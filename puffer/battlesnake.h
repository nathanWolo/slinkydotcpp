// Battlesnake as a PufferLib 3.0 native environment: a C port of slinky's game.
//
// One Battlesnake struct is one game of N snakes; agent slot i is snake i. The rules are a
// stage-by-stage port of slinky's vectorized pipeline (src/slinky/rules.py, maps.py, env.py,
// observations.py), which is itself checked against the official Go engine. The state uses
// slinky's representation (see src/slinky/types.py): (x, y) with (0, 0) bottom-left, grids
// indexed [y][x], and the countdown-grid body (body[i][y][x] = tail pops until the cell is
// vacated: head = length, tail 1, or 2 when stacked).
//
// Random choices follow slinky's distributions, not its random stream, from a per-game PCG32.
#ifndef PUFFER_BATTLESNAKE_H
#define PUFFER_BATTLESNAKE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BS_MAX_SNAKES 16  // the engine's standard and empty maps reject more
#define BS_MAX_DIM 32
#define BS_NUM_ACTIONS 4
#define BS_NUM_CHANNELS 13  // puffer/obs_format.py: CHANNEL_NAMES
#define BS_MASK_SIZE 4
#define BS_MAX_HEALTH 100
#define BS_START_LENGTH 3

enum { BS_UP = 0, BS_DOWN = 1, BS_LEFT = 2, BS_RIGHT = 3 };
static const int BS_DX[BS_NUM_ACTIONS] = {0, 0, -1, 1};
static const int BS_DY[BS_NUM_ACTIONS] = {1, -1, 0, 0};

// slinky.types.Cause
enum {
    CAUSE_NONE = 0,
    CAUSE_OUT_OF_HEALTH = 1,
    CAUSE_OUT_OF_BOUNDS = 2,
    CAUSE_SELF_COLLISION = 3,
    CAUSE_COLLISION = 4,
    CAUSE_HEAD_COLLISION = 5,
    CAUSE_HAZARD = 6,
};

enum {
    RULES_STANDARD,
    RULES_SOLO,
    RULES_WRAPPED,
    RULES_CONSTRICTOR,
    RULES_WRAPPED_CONSTRICTOR,
    RULES_ROYALE,
};
enum { MAP_STANDARD, MAP_EMPTY, MAP_ROYALE };
enum { PLACE_FIXED, PLACE_DISTRIBUTED, PLACE_RANDOM };
enum { REWARD_WIN_LOSS, REWARD_PLACEMENT };

// Per finished game; env_binding.h's vec_log sums every field over games and divides by n.
typedef struct Log {
    float episode_length;  // turns
    float draws;           // nobody survived
    float truncated;       // cut by max_turns
    float deaths_wall;
    float deaths_self;
    float deaths_body;
    float deaths_head;
    float deaths_starve;
    float deaths_hazard;
    float n;
} Log;

typedef struct Battlesnake {
    // PufferLib buffers: this game's num_snakes slots of the shared arrays.
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    Log log;

    // Configuration, fixed at init.
    int width, height, num_snakes, cells;
    int ruleset, map, placement, reward_kind;
    int wrapped, constrictor, solo, royale_rules, has_sides, num_sides;
    int food_spawn_chance, minimum_food, hazard_damage, shrink_every_n_turns, max_turns;
    int compute_obs;
    int obs_rows, obs_cols, board_size, obs_size;

    // Game state: slinky.types.State, field for field.
    int16_t* body;     // [N][H][W]
    int32_t* head;     // [N][2] (x, y); may be off-board for dead snakes
    int32_t* length;   // [N]
    int32_t* health;   // [N]
    uint8_t* alive;    // [N]
    int8_t* last_move; // [N]
    uint8_t* food;     // [H][W]
    int16_t* hazard;   // [H][W] stacked count
    int32_t turn;
    int8_t* elim_cause;  // [N]
    int32_t* elim_turn;  // [N]
    uint8_t done;
    int8_t* sides;  // [W + H - 1] royale shrink sides (RoyaleState); zeros when unused

    // Scratch, allocated once so stepping never allocates.
    uint8_t* mark;     // [H][W]
    int16_t* cand;     // [H][W] candidate cell list
    int16_t* opp_max1; // [H][W] largest living body value on the cell
    int16_t* opp_max2; // [H][W] largest from a snake other than opp_owner
    int8_t* opp_owner; // [H][W] snake holding opp_max1, -1 if none
    uint8_t* hazard8;  // [H][W] saturated hazard count

    uint64_t rng_state, rng_inc;
} Battlesnake;

// --- Random numbers ------------------------------------------------------------------------

static uint64_t bs_splitmix64(uint64_t* x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline uint32_t bs_rand32(Battlesnake* e) {  // PCG32 (XSH RR)
    uint64_t old = e->rng_state;
    e->rng_state = old * 6364136223846793005ULL + e->rng_inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

static void bs_seed(Battlesnake* e, uint64_t seed) {
    uint64_t s = seed;
    e->rng_state = 0;
    e->rng_inc = (bs_splitmix64(&s) << 1u) | 1u;
    bs_rand32(e);
    e->rng_state += bs_splitmix64(&s);
    bs_rand32(e);
}

// Uniform in [0, n), unbiased (Lemire's multiply-and-reject). n >= 1.
static inline uint32_t bs_below(Battlesnake* e, uint32_t n) {
    uint64_t m = (uint64_t)bs_rand32(e) * n;
    uint32_t low = (uint32_t)m;
    if (low < n) {
        uint32_t threshold = (0u - n) % n;
        while (low < threshold) {
            m = (uint64_t)bs_rand32(e) * n;
            low = (uint32_t)m;
        }
    }
    return (uint32_t)(m >> 32);
}

static void bs_shuffle_int(Battlesnake* e, int* xs, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)bs_below(e, (uint32_t)i + 1);
        int t = xs[i];
        xs[i] = xs[j];
        xs[j] = t;
    }
}

// --- Small helpers -------------------------------------------------------------------------

static inline int bs_pmod(int64_t v, int m) {  // Python's %, as jnp's % on int32
    int64_t r = v % m;
    return (int)(r < 0 ? r + m : r);
}

static inline uint8_t bs_sat8(int64_t v) {  // the observation's uint8 saturation
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

static inline int bs_in_bounds(const Battlesnake* e, int64_t x, int64_t y) {
    return x >= 0 && x < e->width && y >= 0 && y < e->height;
}

// slinky.rules.is_game_over: <= 1 snake left (solo, or a 1-snake game: none left).
static int bs_is_game_over(const Battlesnake* e) {
    int remaining = 0;
    for (int i = 0; i < e->num_snakes; i++) remaining += e->alive[i] != 0;
    return e->solo ? remaining == 0 : remaining <= 1;
}

// slinky.rules.royale_hazards: the zone shown at `turn`, after turn // shrink_every_n_turns
// shrinks; shrink k moves edge sides[k] (0 left, 1 right, 2 bottom, 3 top) inward.
static void bs_royale_hazards(Battlesnake* e, int64_t turn) {
    // C division truncates where Python floors, but both leave no shrink for turn < 0.
    int64_t shrinks = turn / e->shrink_every_n_turns;
    int c[4] = {0, 0, 0, 0};
    for (int k = 0; k < e->num_sides && k < shrinks; k++) {
        int s = e->sides[k];
        if (s >= 0 && s < 4) c[s]++;
    }
    const int W = e->width, H = e->height;
    for (int y = 0; y < H; y++) {
        int row_out = y < c[2] || y > H - 1 - c[3];
        for (int x = 0; x < W; x++) {
            e->hazard[y * W + x] = (int16_t)(row_out || x < c[0] || x > W - 1 - c[1]);
        }
    }
}

// --- Rules (slinky.rules.rules_step) -------------------------------------------------------

// One engine ruleset.Execute: move, starvation, hazard damage, feed, elimination, then the
// constrictor and royale stages. No food spawning and no turn increment (the map and the
// env do those). actions[i] outside [0, 4) is an invalid move.
static void bs_rules_step(Battlesnake* e, const int* actions) {
    const int N = e->num_snakes, W = e->width, C = e->cells;
    // The engine's first stage ends the game before anything moves; slinky then returns the
    // state unchanged (the freeze), so nothing below runs.
    if (bs_is_game_over(e)) return;

    uint8_t alive0[BS_MAX_SNAKES], on_board[BS_MAX_SNAKES], hit_hazard[BS_MAX_SNAKES];
    uint8_t eats[BS_MAX_SNAKES], dead[BS_MAX_SNAKES];
    uint8_t hits[BS_MAX_SNAKES][BS_MAX_SNAKES];
    int move[BS_MAX_SNAKES], cell[BS_MAX_SNAKES], cause[BS_MAX_SNAKES], layers[BS_MAX_SNAKES];
    int32_t nx[BS_MAX_SNAKES], ny[BS_MAX_SNAKES];

    // next_heads: invalid actions fall back to the last move; dead snakes don't move.
    for (int i = 0; i < N; i++) {
        alive0[i] = e->alive[i] != 0;
        cause[i] = CAUSE_NONE;
        nx[i] = e->head[2 * i];
        ny[i] = e->head[2 * i + 1];
        move[i] = e->last_move[i];
        cell[i] = -1;
        on_board[i] = 0;
        if (!alive0[i]) continue;
        int a = actions[i];
        move[i] = (a >= 0 && a < BS_NUM_ACTIONS) ? a : e->last_move[i];
        // int32 arithmetic wraps in JAX; the casts keep that (and avoid C overflow).
        nx[i] = (int32_t)((int64_t)nx[i] + BS_DX[move[i]]);
        ny[i] = (int32_t)((int64_t)ny[i] + BS_DY[move[i]]);
        if (e->wrapped) {
            nx[i] = bs_pmod(nx[i], e->width);
            ny[i] = bs_pmod(ny[i], e->height);
        }
        on_board[i] = (uint8_t)bs_in_bounds(e, nx[i], ny[i]);
        if (on_board[i]) cell[i] = ny[i] * W + nx[i];
    }

    // Collisions are judged on the bodies after the tail pop (body_dec in slinky): head i
    // hits snake j if j still occupies the cell, i.e. its countdown there was above 1.
    // Feeding does not change this, so it is read before the body changes.
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            hits[i][j] = cell[i] >= 0 && alive0[j] && e->body[j * C + cell[i]] > 1;
        }
    }

    // Values at the new heads, before feeding removes food (value_at: 0 when off-board).
    uint8_t food_at_head[BS_MAX_SNAKES];
    for (int i = 0; i < N; i++) {
        food_at_head[i] = cell[i] >= 0 && e->food[cell[i]];
        layers[i] = cell[i] >= 0 ? e->hazard[cell[i]] : 0;
    }

    // Move: pop the tail (decrement every countdown), then write the head (= old length).
    for (int i = 0; i < N; i++) {
        if (!alive0[i]) continue;
        int16_t* b = e->body + i * C;
        for (int k = 0; k < C; k++) b[k] = b[k] > 0 ? (int16_t)(b[k] - 1) : 0;
        if (cell[i] >= 0) b[cell[i]] = (int16_t)e->length[i];
    }

    // Starvation.
    for (int i = 0; i < N; i++) {
        if (alive0[i]) e->health[i] = (int32_t)((int64_t)e->health[i] - 1);
    }

    // Hazard damage: skipped if the head's cell also has food. The engine clamps health to
    // [0, 100] after each layer; with one damage value per game every layer moves health the
    // same way, so slinky applies all layers at once and clamps once, saturating the damage
    // to +-100 and the layers to 101 (more change nothing) to stay far from overflow.
    int damage = e->hazard_damage;
    if (damage < -BS_MAX_HEALTH) damage = -BS_MAX_HEALTH;
    if (damage > BS_MAX_HEALTH) damage = BS_MAX_HEALTH;
    uint8_t alive1[BS_MAX_SNAKES];
    for (int i = 0; i < N; i++) {
        hit_hazard[i] = alive0[i] && layers[i] > 0 && !food_at_head[i];
        alive1[i] = alive0[i];
        if (!hit_hazard[i]) continue;
        int64_t l = layers[i] < BS_MAX_HEALTH + 1 ? layers[i] : BS_MAX_HEALTH + 1;
        int64_t h = (int64_t)e->health[i] - l * damage;
        e->health[i] = (int32_t)(h < 0 ? 0 : (h > BS_MAX_HEALTH ? BS_MAX_HEALTH : h));
        if (e->health[i] <= 0) {  // eliminated at once: food later in the turn can't save it
            cause[i] = CAUSE_HAZARD;
            alive1[i] = 0;
        }
    }

    // Feed: every living snake whose head is on food eats it (two heads on one food: both).
    // The tail segment is duplicated, which in countdown form is +1 on every occupied cell.
    for (int i = 0; i < N; i++) eats[i] = alive1[i] && food_at_head[i];
    for (int i = 0; i < N; i++) {
        if (!eats[i]) continue;
        e->food[cell[i]] = 0;
        e->health[i] = BS_MAX_HEALTH;
        e->length[i] = (int32_t)((int64_t)e->length[i] + 1);
        int16_t* b = e->body + i * C;
        for (int k = 0; k < C; k++) if (b[k] > 0) b[k] = (int16_t)(b[k] + 1);
    }

    // Eliminate, pass 1: out of health, then out of bounds. These snakes are gone before
    // collisions are checked, so their bodies can't kill anyone.
    uint8_t alive2[BS_MAX_SNAKES];
    for (int i = 0; i < N; i++) {
        alive2[i] = alive1[i];
        if (!alive1[i]) continue;
        if (e->health[i] <= 0) {
            cause[i] = CAUSE_OUT_OF_HEALTH;
            alive2[i] = 0;
        } else if (!on_board[i]) {
            cause[i] = CAUSE_OUT_OF_BOUNDS;
            alive2[i] = 0;
        }
    }

    // Eliminate, pass 2: self, then another body, then head-to-head (lengths after feeding,
    // `<=` loses), among pass-1 survivors; all results are collected, then applied together.
    for (int i = 0; i < N; i++) {
        if (!alive2[i]) continue;
        if (hits[i][i]) {
            cause[i] = CAUSE_SELF_COLLISION;
            continue;
        }
        int body_hit = 0, h2h = 0;
        for (int j = 0; j < N; j++) {
            if (j == i || !alive2[j]) continue;
            body_hit |= hits[i][j];
            h2h |= nx[i] == nx[j] && ny[i] == ny[j] && e->length[i] <= e->length[j];
        }
        if (body_hit) cause[i] = CAUSE_COLLISION;
        else if (h2h) cause[i] = CAUSE_HEAD_COLLISION;
    }
    uint8_t alive3[BS_MAX_SNAKES];
    for (int i = 0; i < N; i++) {
        alive3[i] = alive2[i] && cause[i] == CAUSE_NONE;
        dead[i] = alive0[i] && !alive3[i];
        if (dead[i]) {
            e->elim_cause[i] = (int8_t)cause[i];
            // eliminatedOnTurn is the turn being resolved plus one.
            e->elim_turn[i] = (int32_t)((int64_t)e->turn + 1);
        }
    }

    // Constrictor: no food, full health (every snake, as in the engine), and grow unless the
    // tail is already stacked (countdown 1 at the tail means a single, unstacked segment).
    if (e->constrictor) {
        memset(e->food, 0, (size_t)C);
        for (int i = 0; i < N; i++) e->health[i] = BS_MAX_HEALTH;
        for (int i = 0; i < N; i++) {
            if (!alive3[i]) continue;
            int16_t* b = e->body + i * C;
            int tail = INT16_MAX;
            for (int k = 0; k < C; k++) if (b[k] > 0 && b[k] < tail) tail = b[k];
            if (tail != 1) continue;
            e->length[i] = (int32_t)((int64_t)e->length[i] + 1);
            for (int k = 0; k < C; k++) if (b[k] > 0) b[k] = (int16_t)(b[k] + 1);
        }
    }

    // Royale: the zone for the next turn, from scratch, after elimination (as the engine's
    // last stage; skipped on the game-over call by the freeze above).
    if (e->royale_rules) bs_royale_hazards(e, (int64_t)e->turn + 1);

    // Eliminated snakes' bodies are not tracked.
    for (int i = 0; i < N; i++) {
        if (!alive3[i]) memset(e->body + i * C, 0, (size_t)C * sizeof(int16_t));
        if (alive0[i]) {
            e->head[2 * i] = nx[i];
            e->head[2 * i + 1] = ny[i];
            e->last_move[i] = (int8_t)move[i];
        }
        e->alive[i] = alive3[i];
    }
}

// --- Maps (slinky.maps) --------------------------------------------------------------------

// spawn_mask: cells free of food and living bodies and (if exclude_head_moves) not
// orthogonally next to a living head; the adjacency does not wrap. Fills e->cand with the
// allowed cells and returns their count.
static int bs_spawn_candidates(Battlesnake* e, int exclude_head_moves) {
    const int N = e->num_snakes, W = e->width, H = e->height, C = e->cells;
    uint8_t* m = e->mark;
    memcpy(m, e->food, (size_t)C);
    for (int i = 0; i < N; i++) {
        if (!e->alive[i]) continue;
        const int16_t* b = e->body + i * C;
        for (int k = 0; k < C; k++) m[k] |= b[k] > 0;
    }
    if (exclude_head_moves) {
        for (int i = 0; i < N; i++) {
            int x = e->head[2 * i], y = e->head[2 * i + 1];
            if (!e->alive[i] || !bs_in_bounds(e, x, y)) continue;
            if (x > 0) m[y * W + x - 1] = 1;
            if (x < W - 1) m[y * W + x + 1] = 1;
            if (y > 0) m[(y - 1) * W + x] = 1;
            if (y < H - 1) m[(y + 1) * W + x] = 1;
        }
    }
    int count = 0;
    for (int k = 0; k < C; k++) if (!m[k]) e->cand[count++] = (int16_t)k;
    return count;
}

// place_food_random: n food on distinct uniformly random spawn_mask cells (fewer if fewer
// are allowed).
static void bs_place_food_random(Battlesnake* e, int n) {
    if (n <= 0) return;
    int count = bs_spawn_candidates(e, 1);
    int k = n < count ? n : count;
    for (int t = 0; t < k; t++) {  // partial Fisher-Yates: a uniform k-subset
        int j = t + (int)bs_below(e, (uint32_t)(count - t));
        int16_t c = e->cand[j];
        e->cand[j] = e->cand[t];
        e->cand[t] = c;
        e->food[c] = 1;
    }
}

// spawn_food_standard: top food up to minimum_food; otherwise one food with the engine's
// probability (chance - 1) / 100 (it tests 100 - rand.Intn(100) < chance).
static void bs_spawn_food_standard(Battlesnake* e) {
    int count = 0;
    for (int k = 0; k < e->cells; k++) count += e->food[k];
    int roll = 100 - (int)bs_below(e, 100);
    int chance = e->food_spawn_chance;
    int need;
    if (count < e->minimum_food) need = e->minimum_food - count;
    else need = (chance > 0 && roll < chance) ? 1 : 0;
    int max_n = e->minimum_food > 1 ? e->minimum_food : 1;
    if (max_n > e->cells) max_n = e->cells;
    bs_place_food_random(e, need < max_n ? need : max_n);
}

// _fixed_food_candidates: the diagonal neighbours of a head allowed for starting food: on
// the far side of the head from the centre, not a corner, not the centre, on the board.
static int bs_fixed_food_allowed(const Battlesnake* e, int hx, int hy, int px, int py) {
    const int W = e->width, H = e->height;
    int cx = (W - 1) / 2, cy = (H - 1) / 2;
    int away = (px < hx && hx < cx) || (cx < hx && hx < px) || (py < hy && hy < cy) ||
               (cy < hy && hy < py);
    int corner = (px == 0 || px == W - 1) && (py == 0 || py == H - 1);
    int centre = px == cx && py == cy;
    return away && !corner && !centre && bs_in_bounds(e, px, py);
}

// place_food_fixed: one food on an allowed diagonal of each head (uniform, skipping cells
// that already have food, in snake order), then one in the centre. On boards smaller than
// 11x11 with more than 4 snakes, only the centre.
static void bs_place_food_fixed(Battlesnake* e) {
    static const int DIAG[4][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    const int W = e->width, H = e->height;
    int small = W * H < 11 * 11;
    if (e->num_snakes <= 4 || !small) {
        for (int i = 0; i < e->num_snakes; i++) {
            int hx = e->head[2 * i], hy = e->head[2 * i + 1];
            int options[4], n = 0;
            for (int d = 0; d < 4; d++) {
                int px = hx + DIAG[d][0], py = hy + DIAG[d][1];
                if (bs_fixed_food_allowed(e, hx, hy, px, py) && !e->food[py * W + px]) {
                    options[n++] = py * W + px;
                }
            }
            if (n > 0) e->food[options[bs_below(e, (uint32_t)n)]] = 1;
        }
    }
    e->food[((H - 1) / 2) * W + (W - 1) / 2] = 1;
}

// placement_kind and _random_start_cells: which of the engine's placements applies, or an
// error message for what slinky rejects.
static const char* bs_placement_kind(int width, int height, int num_snakes, int* kind) {
    if (num_snakes > BS_MAX_SNAKES) return "the standard maps allow at most 16 snakes";
    *kind = PLACE_RANDOM;
    if (width == height) {
        if (num_snakes > 8 && width < 7) return "too many snakes for this board size";
        if (num_snakes <= 8 && width >= 7) {
            *kind = PLACE_FIXED;
            return NULL;
        }
        if (width >= 11) {
            *kind = PLACE_DISTRIBUTED;
            return NULL;
        }
    }
    // Random placement uses cells with even x + y, except the centre.
    int valid = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            valid += (x + y) % 2 == 0 && !(y == (height - 1) / 2 && x == (width - 1) / 2);
        }
    }
    if (valid < num_snakes) return "not enough room to place snakes";
    return NULL;
}

// place_snakes: start points per placement kind, written to e->head.
static void bs_place_snakes(Battlesnake* e) {
    const int N = e->num_snakes, W = e->width, H = e->height;
    if (e->placement == PLACE_FIXED) {
        // Corners and cardinals, each shuffled; corners first or cardinals first, 50/50.
        int mn = 1, md = (W - 1) / 2, mx = W - 2;
        int corners[4][2] = {{mn, mn}, {mn, mx}, {mx, mn}, {mx, mx}};
        int cardinals[4][2] = {{mn, md}, {md, mn}, {md, mx}, {mx, md}};
        int pc[4] = {0, 1, 2, 3}, pd[4] = {0, 1, 2, 3};
        bs_shuffle_int(e, pc, 4);
        bs_shuffle_int(e, pd, 4);
        int corners_first = (int)(bs_rand32(e) & 1u);
        for (int i = 0; i < N; i++) {
            int use_corner = corners_first ? i < 4 : i >= 4;
            const int* p = use_corner ? corners[pc[i % 4]] : cardinals[pd[i % 4]];
            e->head[2 * i] = p[0];
            e->head[2 * i + 1] = p[1];
        }
    } else if (e->placement == PLACE_DISTRIBUTED) {
        // _place_distributed: the quadrants in turn from a random one, each quadrant's 4
        // candidate points in random order.
        int qh = W / 2, qv = H / 2, ho = qh / 3, vo = qv / 3;
        int q0[4][2] = {{ho, vo}, {qh - ho, vo}, {ho, qv - vo}, {qh - ho, qv - vo}};
        int perm[4][4];
        for (int q = 0; q < 4; q++) {
            for (int k = 0; k < 4; k++) perm[q][k] = k;
            bs_shuffle_int(e, perm[q], 4);
        }
        int start = (int)bs_below(e, 4);
        for (int i = 0; i < N; i++) {
            int quad = (start + i) % 4;
            const int* p = q0[perm[quad][i / 4]];
            int x = p[0], y = p[1];
            if (quad == 1 || quad == 3) x = W - 1 - x;
            if (quad == 2 || quad == 3) y = H - 1 - y;
            e->head[2 * i] = x;
            e->head[2 * i + 1] = y;
        }
    } else {
        // _random_cells over _random_start_cells: distinct uniform cells, in random order.
        int count = 0;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                if ((x + y) % 2 == 0 && !(y == (H - 1) / 2 && x == (W - 1) / 2)) {
                    e->cand[count++] = (int16_t)(y * W + x);
                }
            }
        }
        for (int i = 0; i < N; i++) {  // count >= N: checked at init
            int j = i + (int)bs_below(e, (uint32_t)(count - i));
            int16_t c = e->cand[j];
            e->cand[j] = e->cand[i];
            e->cand[i] = c;
            e->head[2 * i] = c % W;
            e->head[2 * i + 1] = c / W;
        }
    }
}

// BattlesnakeEnv.init_state: map setup (snakes, starting food, royale sides), constrictor's
// food removal (the engine runs the ruleset once at setup), and the game-over flag.
static void bs_new_game(Battlesnake* e) {
    const int N = e->num_snakes, C = e->cells;
    memset(e->body, 0, (size_t)N * C * sizeof(int16_t));
    memset(e->food, 0, (size_t)C);
    memset(e->hazard, 0, (size_t)C * sizeof(int16_t));
    memset(e->sides, 0, (size_t)e->num_sides);
    e->turn = 0;
    e->done = 0;
    for (int i = 0; i < N; i++) {
        e->last_move[i] = BS_UP;  // the engine's default move for a stacked snake
        e->elim_cause[i] = CAUSE_NONE;
        e->elim_turn[i] = 0;
    }
    // RoyaleMap and WithRoyaleSides: iid uniform shrink sides, fixed for the game.
    if (e->has_sides) {
        for (int k = 0; k < e->num_sides; k++) e->sides[k] = (int8_t)(bs_rand32(e) >> 30);
    }
    bs_place_snakes(e);
    // snakes_at: fresh length-3 snakes, all segments stacked on the start point.
    for (int i = 0; i < N; i++) {
        e->body[i * C + e->head[2 * i + 1] * e->width + e->head[2 * i]] = BS_START_LENGTH;
        e->length[i] = BS_START_LENGTH;
        e->health[i] = BS_MAX_HEALTH;
        e->alive[i] = 1;
    }
    // place_initial_food (standard and royale maps; the empty map has none).
    if (e->map != MAP_EMPTY) {
        if (e->width == e->height && e->width >= 7) {
            bs_place_food_fixed(e);
        } else {
            bs_place_food_random(e, N < C ? N : C);
        }
    }
    if (e->constrictor) memset(e->food, 0, (size_t)C);
    e->done = (uint8_t)bs_is_game_over(e);
}

// The map's post_update, after the rules and before the turn increment.
static void bs_map_post_update(Battlesnake* e) {
    if (e->map == MAP_EMPTY) return;
    bs_spawn_food_standard(e);
    // RoyaleMap regenerates the zone for the turn about to be shown.
    if (e->map == MAP_ROYALE) bs_royale_hazards(e, (int64_t)e->turn + 1);
}

// --- Environment step (slinky.env.BattlesnakeEnv.step) -------------------------------------

// win_loss_reward / placement_reward for the transition prev_alive -> e->alive.
static void bs_reward(const Battlesnake* e, const uint8_t* prev_alive, float* reward) {
    const int N = e->num_snakes;
    int newly_dead = 0, alive_now = 0;
    for (int i = 0; i < N; i++) {
        newly_dead += prev_alive[i] && !e->alive[i];
        alive_now += e->alive[i] != 0;
    }
    for (int i = 0; i < N; i++) {
        int died = prev_alive[i] && !e->alive[i];
        if (e->solo) {  // nobody to beat or outlast: dying is simply -1
            reward[i] = died ? -1.0f : 0.0f;
        } else if (e->reward_kind == REWARD_WIN_LOSS) {
            // -1 on elimination, +1 to the last snake standing, 0 to all if nobody survives.
            int over = bs_is_game_over(e);
            int draw = over && alive_now == 0;
            reward[i] = (died && !draw) ? -1.0f : 0.0f;
            if (over && e->alive[i]) reward[i] = 1.0f;
        } else {
            // 1 / (N - 1) per snake outlasted this turn, minus 1 / (N - 1) per survivor.
            int gain = e->alive[i] ? newly_dead : 0;
            int loss = died ? alive_now : 0;
            reward[i] = (float)(gain - loss) / (float)(N - 1);
        }
    }
}

// One full game step without autoreset. Stepping a finished game is a no-op with zero
// reward. Sets *truncated when max_turns (not the rules) ended the game on this step.
static void bs_game_step(Battlesnake* e, const int* actions, float* reward, int* truncated) {
    const int N = e->num_snakes;
    *truncated = 0;
    if (e->done) {
        for (int i = 0; i < N; i++) reward[i] = 0.0f;
        return;
    }
    uint8_t prev_alive[BS_MAX_SNAKES];
    memcpy(prev_alive, e->alive, (size_t)N);
    bs_rules_step(e, actions);
    bs_map_post_update(e);
    e->turn = (int32_t)((int64_t)e->turn + 1);
    int over = bs_is_game_over(e);
    *truncated = !over && e->max_turns > 0 && e->turn >= e->max_turns;
    e->done = (uint8_t)(over || *truncated);
    bs_reward(e, prev_alive, reward);
}

// --- Observations (slinky.observations.egocentric_obs, slinky.rules.action_mask) ----------

// certainly_starving: on 1 health with no food (or healing hazard) next to the head.
static void bs_certainly_starving(const Battlesnake* e, uint8_t* starving) {
    const int W = e->width;
    for (int i = 0; i < e->num_snakes; i++) {
        starving[i] = 0;
        if (!e->alive[i] || e->health[i] > 1) continue;
        int rescue = 0;
        for (int a = 0; a < BS_NUM_ACTIONS; a++) {
            int64_t x = (int64_t)e->head[2 * i] + BS_DX[a];
            int64_t y = (int64_t)e->head[2 * i + 1] + BS_DY[a];
            if (e->wrapped) {
                x = bs_pmod(x, e->width);
                y = bs_pmod(y, e->height);
            }
            if (!bs_in_bounds(e, x, y)) continue;
            int k = (int)(y * W + x);
            rescue |= e->food[k] || (e->hazard_damage < 0 && e->hazard[k] > 0);
        }
        starving[i] = !rescue;
    }
}

// action_mask: moves that don't certainly die to a wall or to a cell that holds a body next
// turn whatever anyone does (countdown above 1, bodies of certainly-starving snakes not
// counted). Dead, certainly-starving and trapped snakes get all-ones rows.
static void bs_action_mask(const Battlesnake* e, uint8_t mask[][BS_MASK_SIZE]) {
    const int N = e->num_snakes, W = e->width, C = e->cells;
    uint8_t starving[BS_MAX_SNAKES], solid[BS_MAX_SNAKES];
    bs_certainly_starving(e, starving);
    for (int j = 0; j < N; j++) solid[j] = e->alive[j] && !starving[j];
    for (int i = 0; i < N; i++) {
        int any_ok = 0;
        for (int a = 0; a < BS_NUM_ACTIONS; a++) {
            int64_t x = (int64_t)e->head[2 * i] + BS_DX[a];
            int64_t y = (int64_t)e->head[2 * i + 1] + BS_DY[a];
            if (e->wrapped) {
                x = bs_pmod(x, e->width);
                y = bs_pmod(y, e->height);
            }
            int ok = bs_in_bounds(e, x, y);
            if (ok) {
                int k = (int)(y * W + x);
                for (int j = 0; j < N && ok; j++) ok = !(solid[j] && e->body[j * C + k] > 1);
            }
            mask[i][a] = (uint8_t)ok;
            any_ok |= ok;
        }
        if (!e->alive[i] || starving[i] || !any_ok) {
            for (int a = 0; a < BS_NUM_ACTIONS; a++) mask[i][a] = 1;
        }
    }
}

// Every slot's observation of the current state: the egocentric board as raw integers
// (puffer/obs_format.py), then the action mask. Dead slots: zero board, all-ones mask.
static void bs_observe(Battlesnake* e) {
    const int N = e->num_snakes, W = e->width, H = e->height, C = e->cells;
    const int cols = e->obs_cols;
    uint8_t mask[BS_MAX_SNAKES][BS_MASK_SIZE];
    bs_action_mask(e, mask);

    // Per-cell aggregates over living snakes, once for all agents. opp_countdown for agent
    // a is the max over living j != a of body[j] (and 0, from a's own masked entry), which
    // is opp_max1 unless a holds it, then opp_max2. opp_body is that value > 0.
    for (int k = 0; k < C; k++) {
        e->opp_max1[k] = 0;
        e->opp_max2[k] = 0;
        e->opp_owner[k] = -1;
        e->hazard8[k] = bs_sat8(e->hazard[k]);
    }
    for (int j = 0; j < N; j++) {
        if (!e->alive[j]) continue;
        const int16_t* b = e->body + j * C;
        for (int k = 0; k < C; k++) {
            int16_t v = b[k];
            if (v > e->opp_max1[k]) {
                e->opp_max2[k] = e->opp_max1[k];
                e->opp_max1[k] = v;
                e->opp_owner[k] = (int8_t)j;
            } else if (v > e->opp_max2[k]) {
                e->opp_max2[k] = v;
            }
        }
    }

    for (int i = 0; i < N; i++) {
        uint8_t* out = e->observations + (size_t)i * e->obs_size;
        uint8_t* m = out + e->board_size;
        if (!e->alive[i]) {
            memset(out, 0, (size_t)e->board_size);
            memset(m, 1, BS_MASK_SIZE);
            continue;
        }
        memcpy(m, mask[i], BS_MASK_SIZE);
        // Window placement. Non-wrapped: cell (x, y) at row y + H-1-hy, col x + W-1-hx of
        // the zero-padded [2H-1, 2W-1] image (head clamped, as slinky does). Wrapped: the
        // board rolled so the head sits at [H//2, W//2].
        int hx = e->head[2 * i], hy = e->head[2 * i + 1];
        hx = hx < 0 ? 0 : (hx > W - 1 ? W - 1 : hx);
        hy = hy < 0 ? 0 : (hy > H - 1 ? H - 1 : hy);
        int row0, col0;
        if (e->wrapped) {
            row0 = H / 2 - hy;
            col0 = W / 2 - hx;
        } else {
            memset(out, 0, (size_t)e->board_size);
            row0 = H - 1 - hy;
            col0 = W - 1 - hx;
        }
        // Locals, not e->...: stores through uint8_t* may alias the struct, which would
        // make the compiler reload every pointer after every byte written.
        const int16_t* restrict own = e->body + i * C;
        const uint8_t* restrict food = e->food;
        const uint8_t* restrict hazard8 = e->hazard8;
        const int16_t* restrict max1 = e->opp_max1;
        const int16_t* restrict max2 = e->opp_max2;
        const int8_t* restrict owner = e->opp_owner;
        const uint8_t health8 = bs_sat8(e->health[i]), length8 = bs_sat8(e->length[i]);
        size_t cell_at[BS_MAX_DIM];  // byte offset of board column x within an output row
        for (int x = 0; x < W; x++) {
            int c = x + col0;
            if (e->wrapped) c = c >= W ? c - W : (c < 0 ? c + W : c);
            cell_at[x] = (size_t)c * BS_NUM_CHANNELS;
        }
        for (int y = 0; y < H; y++) {
            int r = y + row0;
            if (e->wrapped) r = r >= H ? r - H : (r < 0 ? r + H : r);
            uint8_t* restrict orow = out + (size_t)r * cols * BS_NUM_CHANNELS;
            for (int x = 0; x < W; x++) {
                int k = y * W + x;
                int own_v = own[k];
                int opp_v = owner[k] == i ? max2[k] : max1[k];
                uint8_t* restrict o = orow + cell_at[x];
                o[0] = 1;  // on_board
                o[1] = food[k];
                o[2] = hazard8[k];
                o[3] = 0;  // own_head, below
                o[4] = own_v > 0;
                o[5] = bs_sat8(own_v);
                o[6] = 0;  // opp_head_ge / opp_head_lt, below
                o[7] = 0;
                o[8] = opp_v > 0;
                o[9] = bs_sat8(opp_v);
                o[10] = 0;  // opp_head_health, below
                o[11] = health8;
                o[12] = length8;
            }
        }
        // Heads: one-hot on the real (unclamped) position, nothing when off-board.
        for (int j = 0; j < N; j++) {
            if (!e->alive[j]) continue;
            int x = e->head[2 * j], y = e->head[2 * j + 1];
            if (!bs_in_bounds(e, x, y)) continue;
            int r = y + row0, c = x + col0;
            if (e->wrapped) {
                r = r >= H ? r - H : (r < 0 ? r + H : r);
                c = c >= W ? c - W : (c < 0 ? c + W : c);
            }
            uint8_t* o = out + ((size_t)r * cols + c) * BS_NUM_CHANNELS;
            if (j == i) {
                o[3] = 1;
                continue;
            }
            if (e->length[j] >= e->length[i]) o[6] = 1;
            else o[7] = 1;
            uint8_t h = bs_sat8(e->health[j]);  // max over opponents' heads, and 0
            if (h > o[10]) o[10] = h;
        }
    }
}

// --- PufferLib interface -------------------------------------------------------------------

static void bs_add_log(Battlesnake* e, int truncated) {
    int alive_now = 0;
    for (int i = 0; i < e->num_snakes; i++) {
        alive_now += e->alive[i] != 0;
        if (e->alive[i]) continue;
        switch (e->elim_cause[i]) {
            case CAUSE_OUT_OF_BOUNDS: e->log.deaths_wall += 1; break;
            case CAUSE_SELF_COLLISION: e->log.deaths_self += 1; break;
            case CAUSE_COLLISION: e->log.deaths_body += 1; break;
            case CAUSE_HEAD_COLLISION: e->log.deaths_head += 1; break;
            case CAUSE_OUT_OF_HEALTH: e->log.deaths_starve += 1; break;
            case CAUSE_HAZARD: e->log.deaths_hazard += 1; break;
            default: break;
        }
    }
    e->log.episode_length += (float)e->turn;
    e->log.draws += alive_now == 0;
    e->log.truncated += truncated != 0;
    e->log.n += 1;
}

// Slot semantics for PuffeRL, which stores (obs_t, action_t, value_t, reward_t, terminal_t)
// with reward_t and terminal_t returned together with obs_t, and bootstraps step t from
// r_{t+1} and (1 - d_{t+1}): a slot is terminal when its snake died on this transition, when
// the game ended (rules or max_turns; PufferLib 3.0 drops truncations), or when the snake was
// already dead (one-step episodes with reward 0 that carry no signal). A finished game is
// logged and replaced at once, and the observations written are the new game's.
static void c_step(Battlesnake* e) {
    const int N = e->num_snakes;
    uint8_t prev_alive[BS_MAX_SNAKES];
    memcpy(prev_alive, e->alive, (size_t)N);
    int was_done = e->done, truncated;
    bs_game_step(e, e->actions, e->rewards, &truncated);
    for (int i = 0; i < N; i++) e->terminals[i] = !prev_alive[i] || !e->alive[i] || e->done;
    if (e->done) {
        if (!was_done) bs_add_log(e, truncated);
        bs_new_game(e);
    }
    if (e->compute_obs) bs_observe(e);
}

// A new game. vec_reset seeds rand() per env before calling this, so reseeding from it makes
// reset(seed) reproducible.
static void c_reset(Battlesnake* e) {
    uint64_t s = (uint64_t)(unsigned)rand();
    s = (s << 31) ^ (uint64_t)(unsigned)rand();
    bs_seed(e, s);
    bs_new_game(e);
    memset(e->rewards, 0, (size_t)e->num_snakes * sizeof(float));
    memset(e->terminals, 0, (size_t)e->num_snakes);
    bs_observe(e);
}

static int bs_digits(int v) {
    int d = 1;
    while (v >= 10) {
        v /= 10;
        d++;
    }
    return d;
}

// The ASCII board of slinky.render.render: top row y = H-1, heads 'A' + i, bodies 'a' + i,
// '*' food, '~' hazard on otherwise empty cells, '.' empty; dead snakes listed below.
static void c_render(Battlesnake* e) {
    static const char* CAUSE_NAMES[] = {"NONE", "OUT_OF_HEALTH", "OUT_OF_BOUNDS",
                                        "SELF_COLLISION", "COLLISION", "HEAD_COLLISION",
                                        "HAZARD"};
    const int N = e->num_snakes, W = e->width, H = e->height, C = e->cells;
    char grid[BS_MAX_DIM * BS_MAX_DIM];
    for (int k = 0; k < C; k++) grid[k] = e->hazard[k] > 0 ? '~' : '.';
    for (int k = 0; k < C; k++) if (e->food[k]) grid[k] = '*';
    for (int i = 0; i < N; i++) {  // bodies first so that heads always win
        if (!e->alive[i]) continue;
        for (int k = 0; k < C; k++) if (e->body[i * C + k] > 0) grid[k] = (char)('a' + i);
    }
    for (int i = 0; i < N; i++) {
        int x = e->head[2 * i], y = e->head[2 * i + 1];
        if (e->alive[i] && bs_in_bounds(e, x, y)) grid[y * W + x] = (char)('A' + i);
    }
    int cw = bs_digits(W - 1), yw = bs_digits(H - 1);
    printf("turn %d%s\n", e->turn, e->done ? "  (game over)" : "");
    for (int y = H - 1; y >= 0; y--) {
        printf("%*d", yw, y);
        for (int x = 0; x < W; x++) printf(" %*c", cw, grid[y * W + x]);
        printf("\n");
    }
    printf("%*s", yw + 1, "");
    for (int x = 0; x < W; x++) printf(x ? " %*d" : "%*d", cw, x);
    printf("\n");
    for (int i = 0; i < N; i++) {
        printf("%c ", 'A' + i);
        int cause = e->elim_cause[i];
        if (e->alive[i]) printf("alive");
        else if (cause >= 0 && cause <= CAUSE_HAZARD) printf("dead (%s)", CAUSE_NAMES[cause]);
        else printf("dead (%d)", cause);
        printf(" len=%d health=%d\n", e->length[i], e->health[i]);
    }
    fflush(stdout);
}

static void c_close(Battlesnake* e) {
    void* blocks[] = {e->body, e->head, e->length, e->health, e->alive, e->last_move, e->food,
                      e->hazard, e->elim_cause, e->elim_turn, e->sides, e->mark, e->cand,
                      e->opp_max1, e->opp_max2, e->opp_owner, e->hazard8};
    for (size_t k = 0; k < sizeof(blocks) / sizeof(blocks[0]); k++) free(blocks[k]);
    e->body = NULL;
    e->head = e->length = e->health = e->elim_turn = NULL;
    e->alive = e->food = e->mark = e->hazard8 = NULL;
    e->last_move = e->elim_cause = e->sides = e->opp_owner = NULL;
    e->hazard = e->cand = e->opp_max1 = e->opp_max2 = NULL;
}

// Configure and allocate one game. Returns NULL, or an error message (nothing allocated).
static const char* bs_init(Battlesnake* e, int width, int height, int num_snakes, int ruleset,
                           int map, int food_spawn_chance, int minimum_food, int hazard_damage,
                           int shrink_every_n_turns, int max_turns, int reward_kind,
                           int compute_obs, uint64_t seed) {
    if (width < 1 || width > BS_MAX_DIM || height < 1 || height > BS_MAX_DIM) {
        return "width and height must be in [1, 32]";
    }
    if (num_snakes < 1 || num_snakes > BS_MAX_SNAKES) return "num_snakes must be in [1, 16]";
    if (shrink_every_n_turns < 1) return "shrink_every_n_turns must be at least 1";
    if (max_turns < 0) return "max_turns must be >= 0 (0: no limit)";
    int placement;
    const char* err = bs_placement_kind(width, height, num_snakes, &placement);
    if (err) return err;

    e->width = width;
    e->height = height;
    e->num_snakes = num_snakes;
    e->cells = width * height;
    e->ruleset = ruleset;
    e->map = map;
    e->placement = placement;
    e->reward_kind = reward_kind;
    e->wrapped = ruleset == RULES_WRAPPED || ruleset == RULES_WRAPPED_CONSTRICTOR;
    e->constrictor = ruleset == RULES_CONSTRICTOR || ruleset == RULES_WRAPPED_CONSTRICTOR;
    e->solo = ruleset == RULES_SOLO || num_snakes == 1;
    e->royale_rules = ruleset == RULES_ROYALE;
    e->has_sides = e->royale_rules || map == MAP_ROYALE;
    e->num_sides = width + height - 1;
    e->food_spawn_chance = food_spawn_chance;
    e->minimum_food = minimum_food;
    e->hazard_damage = hazard_damage;
    e->shrink_every_n_turns = shrink_every_n_turns;
    e->max_turns = max_turns;
    e->compute_obs = compute_obs;
    e->obs_rows = e->wrapped ? height : 2 * height - 1;
    e->obs_cols = e->wrapped ? width : 2 * width - 1;
    e->board_size = e->obs_rows * e->obs_cols * BS_NUM_CHANNELS;
    e->obs_size = e->board_size + BS_MASK_SIZE;

    size_t N = (size_t)num_snakes, C = (size_t)e->cells;
    e->body = calloc(N * C, sizeof(int16_t));
    e->head = calloc(2 * N, sizeof(int32_t));
    e->length = calloc(N, sizeof(int32_t));
    e->health = calloc(N, sizeof(int32_t));
    e->alive = calloc(N, 1);
    e->last_move = calloc(N, 1);
    e->food = calloc(C, 1);
    e->hazard = calloc(C, sizeof(int16_t));
    e->elim_cause = calloc(N, 1);
    e->elim_turn = calloc(N, sizeof(int32_t));
    e->sides = calloc((size_t)e->num_sides, 1);
    e->mark = calloc(C, 1);
    e->cand = calloc(C, sizeof(int16_t));
    e->opp_max1 = calloc(C, sizeof(int16_t));
    e->opp_max2 = calloc(C, sizeof(int16_t));
    e->opp_owner = calloc(C, 1);
    e->hazard8 = calloc(C, 1);
    if (!e->body || !e->head || !e->length || !e->health || !e->alive || !e->last_move ||
        !e->food || !e->hazard || !e->elim_cause || !e->elim_turn || !e->sides || !e->mark ||
        !e->cand || !e->opp_max1 || !e->opp_max2 || !e->opp_owner || !e->hazard8) {
        c_close(e);
        return "out of memory";
    }
    bs_seed(e, seed);
    bs_new_game(e);
    return NULL;
}

#endif  // PUFFER_BATTLESNAKE_H
