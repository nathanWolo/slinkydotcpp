// PufferLib 3.0 binding of the Battlesnake environment (puffer/battlesnake.h).
//
// env_binding.h supplies the standard env_* / vec_* entry points; this file adds my_init,
// my_log, the state accessors env_get / env_put (MY_GET / MY_PUT) and the test hooks in
// MY_METHODS, which drive one game through the same functions c_step and c_reset use.
#define PY_SSIZE_T_CLEAN
#include <Python.h>  // before any standard header, as Python requires

#include "battlesnake.h"

#define Env Battlesnake
#define MY_GET
#define MY_PUT

static PyObject* py_rules_step(PyObject* self, PyObject* args);
static PyObject* py_game_step(PyObject* self, PyObject* args);
static PyObject* py_observe(PyObject* self, PyObject* args);
static PyObject* py_new_game(PyObject* self, PyObject* args);
static PyObject* py_seed(PyObject* self, PyObject* args);

#define MY_METHODS                                                                          \
    {"rules_step", py_rules_step, METH_VARARGS,                                             \
     "rules_step(h, actions): slinky.rules.rules_step on the game's state"},                \
        {"game_step", py_game_step, METH_VARARGS,                                           \
         "game_step(h, actions) -> float32[N]: one BattlesnakeEnv.step, no autoreset"},     \
        {"observe", py_observe, METH_VARARGS,                                               \
         "observe(h): write the current state's observations to the env's buffer"},         \
        {"new_game", py_new_game, METH_VARARGS,                                             \
         "new_game(h): a fresh turn-0 game from the env's PRNG (no observations)"},         \
        {"seed", py_seed, METH_VARARGS, "seed(h, s): reseed the env's PRNG"}

// env_binding.h is PufferLib's file; keep its own warnings out of this build's.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wunused-function"
#include "env_binding.h"
#pragma GCC diagnostic pop

// --- Keyword arguments -----------------------------------------------------------------------

// An optional int keyword (bools are ints); `fallback` when absent. -1 with an error set.
static int kw_int(PyObject* kwargs, const char* key, long fallback, int* out) {
    PyObject* v = kwargs ? PyDict_GetItemString(kwargs, key) : NULL;
    if (v == NULL) {
        *out = (int)fallback;
        return 0;
    }
    if (!PyLong_Check(v)) {
        PyErr_Format(PyExc_TypeError, "%s must be an int", key);
        return -1;
    }
    long x = PyLong_AsLong(v);
    if (PyErr_Occurred()) return -1;
    if (x < INT_MIN || x > INT_MAX) {
        PyErr_Format(PyExc_ValueError, "%s=%ld is out of range", key, x);
        return -1;
    }
    *out = (int)x;
    return 0;
}

// An optional string keyword, mapped to its index in `names` (NULL-terminated).
static int kw_choice(PyObject* kwargs, const char* key, const char* const* names, int* out) {
    PyObject* v = kwargs ? PyDict_GetItemString(kwargs, key) : NULL;
    if (v == NULL) {
        *out = 0;
        return 0;
    }
    if (!PyUnicode_Check(v)) {
        PyErr_Format(PyExc_TypeError, "%s must be a str", key);
        return -1;
    }
    const char* s = PyUnicode_AsUTF8(v);
    if (s == NULL) return -1;
    for (int k = 0; names[k]; k++) {
        if (strcmp(s, names[k]) == 0) {
            *out = k;
            return 0;
        }
    }
    PyErr_Format(PyExc_ValueError, "unknown %s %R", key, v);
    return -1;
}

static const char* const RULESET_NAMES[] = {"standard",    "solo",
                                            "wrapped",     "constrictor",
                                            "wrapped_constrictor", "royale",
                                            NULL};
static const char* const MAP_NAMES[] = {"standard", "empty", "royale", NULL};
static const char* const REWARD_NAMES[] = {"win_loss", "placement", NULL};

// Keywords (defaults as slinky.types.GameConfig): width, height, num_snakes, ruleset, map,
// food_spawn_chance, minimum_food, hazard_damage_per_turn, shrink_every_n_turns, max_turns
// (0: no limit), reward ("win_loss" or "placement"), compute_obs, and the seed env_binding.h
// adds.
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    (void)args;
    int width, height, num_snakes, ruleset, map, chance, min_food, damage, shrink, max_turns;
    int reward, compute_obs, seed;
    if (kw_int(kwargs, "width", 11, &width) || kw_int(kwargs, "height", 11, &height) ||
        kw_int(kwargs, "num_snakes", 2, &num_snakes) ||
        kw_choice(kwargs, "ruleset", RULESET_NAMES, &ruleset) ||
        kw_choice(kwargs, "map", MAP_NAMES, &map) ||
        kw_int(kwargs, "food_spawn_chance", 15, &chance) ||
        kw_int(kwargs, "minimum_food", 1, &min_food) ||
        kw_int(kwargs, "hazard_damage_per_turn", 14, &damage) ||
        kw_int(kwargs, "shrink_every_n_turns", 25, &shrink) ||
        kw_int(kwargs, "max_turns", 0, &max_turns) ||
        kw_choice(kwargs, "reward", REWARD_NAMES, &reward) ||
        kw_int(kwargs, "compute_obs", 1, &compute_obs) || kw_int(kwargs, "seed", 0, &seed)) {
        return -1;
    }
    const char* err = bs_init(env, width, height, num_snakes, ruleset, map, chance, min_food,
                              damage, shrink, max_turns, reward, compute_obs != 0,
                              (uint64_t)(uint32_t)seed);
    if (err) {
        PyErr_SetString(PyExc_ValueError, err);
        return -1;
    }
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    // vec_log adds "n" itself.
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "draws", log->draws);
    assign_to_dict(dict, "truncated", log->truncated);
    assign_to_dict(dict, "deaths_wall", log->deaths_wall);
    assign_to_dict(dict, "deaths_self", log->deaths_self);
    assign_to_dict(dict, "deaths_body", log->deaths_body);
    assign_to_dict(dict, "deaths_head", log->deaths_head);
    assign_to_dict(dict, "deaths_starve", log->deaths_starve);
    assign_to_dict(dict, "deaths_hazard", log->deaths_hazard);
    return 0;
}

// --- State access (slinky.types.State field names and dtypes) --------------------------------

typedef struct {
    const char* name;
    int type;  // NPY_*
    int ndim;
    npy_intp dims[3];
    void* data;
} Field;

// The array fields of the state; turn and done are scalars, handled separately.
static int state_fields(Env* e, Field* f) {
    npy_intp N = e->num_snakes, H = e->height, W = e->width;
    Field fields[] = {
        {"body", NPY_INT16, 3, {N, H, W}, e->body},
        {"head", NPY_INT32, 2, {N, 2, 0}, e->head},
        {"length", NPY_INT32, 1, {N, 0, 0}, e->length},
        {"health", NPY_INT32, 1, {N, 0, 0}, e->health},
        {"alive", NPY_BOOL, 1, {N, 0, 0}, e->alive},
        {"last_move", NPY_INT8, 1, {N, 0, 0}, e->last_move},
        {"food", NPY_BOOL, 2, {H, W, 0}, e->food},
        {"hazard", NPY_INT16, 2, {H, W, 0}, e->hazard},
        {"elim_cause", NPY_INT8, 1, {N, 0, 0}, e->elim_cause},
        {"elim_turn", NPY_INT32, 1, {N, 0, 0}, e->elim_turn},
        {"sides", NPY_INT8, 1, {e->num_sides, 0, 0}, e->sides},
    };
    int n = (int)(sizeof(fields) / sizeof(fields[0]));
    memcpy(f, fields, sizeof(fields));
    return n;
}
#define NUM_ARRAY_FIELDS 11

static int set_item(PyObject* dict, const char* key, PyObject* value) {
    if (value == NULL) return -1;
    int rc = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    return rc;
}

// env_get(h) -> dict of copies; turn is a 0-d int32 array and done a 0-d bool array.
static PyObject* my_get(PyObject* dict, Env* e) {
    Field f[NUM_ARRAY_FIELDS];
    int n = state_fields(e, f);
    for (int k = 0; k < n; k++) {
        PyObject* a = PyArray_SimpleNew(f[k].ndim, f[k].dims, f[k].type);
        if (a == NULL) return NULL;
        memcpy(PyArray_DATA((PyArrayObject*)a), f[k].data, PyArray_NBYTES((PyArrayObject*)a));
        if (set_item(dict, f[k].name, a) < 0) return NULL;
    }
    PyObject* turn = PyArray_SimpleNew(0, NULL, NPY_INT32);
    if (turn == NULL) return NULL;
    *(int32_t*)PyArray_DATA((PyArrayObject*)turn) = e->turn;
    if (set_item(dict, "turn", turn) < 0) return NULL;
    PyObject* done = PyArray_SimpleNew(0, NULL, NPY_BOOL);
    if (done == NULL) return NULL;
    *(npy_bool*)PyArray_DATA((PyArrayObject*)done) = e->done != 0;
    if (set_item(dict, "done", done) < 0) return NULL;
    return dict;
}

static int check_array(PyObject* v, const Field* f) {
    if (!PyArray_Check(v)) {
        PyErr_Format(PyExc_TypeError, "%s must be a numpy array", f->name);
        return -1;
    }
    PyArrayObject* a = (PyArrayObject*)v;
    PyArray_Descr* want = PyArray_DescrFromType(f->type);
    int same = PyArray_EquivTypes(PyArray_DESCR(a), want);
    Py_DECREF(want);
    if (!same) {
        PyErr_Format(PyExc_TypeError, "%s must have dtype %s, got %R", f->name,
                     f->type == NPY_BOOL ? "bool" : (f->type == NPY_INT8 ? "int8" :
                     (f->type == NPY_INT16 ? "int16" : "int32")), (PyObject*)PyArray_DESCR(a));
        return -1;
    }
    int ok = PyArray_NDIM(a) == f->ndim;
    for (int d = 0; ok && d < f->ndim; d++) ok = PyArray_DIM(a, d) == f->dims[d];
    if (!ok) {
        PyErr_Format(PyExc_ValueError, "%s has the wrong shape", f->name);
        return -1;
    }
    return 0;
}

// A scalar field: a Python int / bool, a numpy scalar or a 0-d array of the right kind.
static int read_scalar(PyObject* v, const char* name, int is_bool, int64_t* out) {
    PyArrayObject* a = (PyArrayObject*)PyArray_FROM_O(v);
    if (a == NULL) return -1;
    int kind = PyArray_DESCR(a)->kind;
    int ok = PyArray_NDIM(a) == 0 && (is_bool ? kind == 'b' : (kind == 'i' || kind == 'u'));
    if (!ok) {
        Py_DECREF(a);
        PyErr_Format(PyExc_TypeError, "%s must be a %s scalar", name, is_bool ? "bool" : "int");
        return -1;
    }
    PyObject* item = PyArray_ToScalar(PyArray_DATA(a), a);
    Py_DECREF(a);
    if (item == NULL) return -1;
    long long x = is_bool ? PyObject_IsTrue(item) : PyLong_AsLongLong(item);
    Py_DECREF(item);
    if (x == -1 && PyErr_Occurred()) return -1;
    if (!is_bool && (x < INT32_MIN || x > INT32_MAX)) {
        PyErr_Format(PyExc_ValueError, "%s is out of the int32 range", name);
        return -1;
    }
    *out = x;
    return 0;
}

// env_put(h, **fields): every field of env_get, validated before anything is written.
static int my_put(Env* e, PyObject* args, PyObject* kwargs) {
    (void)args;
    Field f[NUM_ARRAY_FIELDS];
    int n = state_fields(e, f);
    if (kwargs == NULL) {
        PyErr_SetString(PyExc_TypeError, "env_put needs the state fields as keywords");
        return -1;
    }
    Py_ssize_t pos = 0;
    PyObject *key, *value;
    while (PyDict_Next(kwargs, &pos, &key, &value)) {
        const char* name = PyUnicode_AsUTF8(key);
        if (name == NULL) return -1;
        int known = strcmp(name, "turn") == 0 || strcmp(name, "done") == 0;
        for (int k = 0; !known && k < n; k++) known = strcmp(name, f[k].name) == 0;
        if (!known) {
            PyErr_Format(PyExc_TypeError, "unknown state field %R", key);
            return -1;
        }
    }
    PyObject* values[NUM_ARRAY_FIELDS];
    for (int k = 0; k < n; k++) {
        values[k] = PyDict_GetItemString(kwargs, f[k].name);
        if (values[k] == NULL) {
            PyErr_Format(PyExc_TypeError, "missing state field '%s'", f[k].name);
            return -1;
        }
        if (check_array(values[k], &f[k]) < 0) return -1;
    }
    int64_t turn, done;
    PyObject* t = PyDict_GetItemString(kwargs, "turn");
    PyObject* d = PyDict_GetItemString(kwargs, "done");
    if (t == NULL || d == NULL) {
        PyErr_Format(PyExc_TypeError, "missing state field '%s'", t == NULL ? "turn" : "done");
        return -1;
    }
    if (read_scalar(t, "turn", 0, &turn) < 0 || read_scalar(d, "done", 1, &done) < 0) return -1;
    // last_move indexes the move table (an invalid action repeats it): slinky's states only
    // ever hold real moves there.
    PyArrayObject* lm = (PyArrayObject*)values[5];
    for (npy_intp i = 0; i < PyArray_DIM(lm, 0); i++) {
        int8_t m = *(int8_t*)PyArray_GETPTR1(lm, i);
        if (m < 0 || m >= BS_NUM_ACTIONS) {
            PyErr_SetString(PyExc_ValueError, "last_move must be in [0, 4)");
            return -1;
        }
    }
    for (int k = 0; k < n; k++) {
        // Copies through a contiguous array of the field's type (bools as 0/1 bytes).
        PyArrayObject* a = (PyArrayObject*)PyArray_FROM_OTF(values[k], f[k].type,
                                                            NPY_ARRAY_IN_ARRAY);
        if (a == NULL) return -1;
        memcpy(f[k].data, PyArray_DATA(a), PyArray_NBYTES(a));
        Py_DECREF(a);
    }
    for (int k = 0; k < e->cells; k++) e->food[k] = e->food[k] != 0;
    for (int i = 0; i < e->num_snakes; i++) e->alive[i] = e->alive[i] != 0;
    e->turn = (int32_t)turn;
    e->done = done != 0;
    return 0;
}

// --- Test hooks ----------------------------------------------------------------------------

static Env* hook_env(PyObject* args, Py_ssize_t nargs, const char* name) {
    if (PyTuple_Size(args) != nargs) {
        PyErr_Format(PyExc_TypeError, "%s takes %zd argument(s)", name, nargs);
        return NULL;
    }
    return unpack_env(args);
}

// int[N] actions from any integer sequence; values outside [0, 4) become -1 (invalid).
static int read_actions(Env* e, PyObject* obj, int* actions) {
    PyArrayObject* a = (PyArrayObject*)PyArray_FROMANY(obj, NPY_INT64, 1, 1,
                                                       NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST);
    if (a == NULL) return -1;
    if (PyArray_DIM(a, 0) != e->num_snakes) {
        Py_DECREF(a);
        PyErr_Format(PyExc_ValueError, "actions must have shape (%d,)", e->num_snakes);
        return -1;
    }
    const int64_t* v = (const int64_t*)PyArray_DATA(a);
    for (int i = 0; i < e->num_snakes; i++) {
        actions[i] = (v[i] >= 0 && v[i] < BS_NUM_ACTIONS) ? (int)v[i] : -1;
    }
    Py_DECREF(a);
    return 0;
}

static PyObject* py_rules_step(PyObject* self, PyObject* args) {
    (void)self;
    Env* e = hook_env(args, 2, "rules_step");
    int actions[BS_MAX_SNAKES];
    if (e == NULL || read_actions(e, PyTuple_GetItem(args, 1), actions) < 0) return NULL;
    bs_rules_step(e, actions);
    Py_RETURN_NONE;
}

static PyObject* py_game_step(PyObject* self, PyObject* args) {
    (void)self;
    Env* e = hook_env(args, 2, "game_step");
    int actions[BS_MAX_SNAKES];
    if (e == NULL || read_actions(e, PyTuple_GetItem(args, 1), actions) < 0) return NULL;
    npy_intp dims[1] = {e->num_snakes};
    PyObject* rewards = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (rewards == NULL) return NULL;
    int truncated;
    bs_game_step(e, actions, (float*)PyArray_DATA((PyArrayObject*)rewards), &truncated);
    return rewards;
}

static PyObject* py_observe(PyObject* self, PyObject* args) {
    (void)self;
    Env* e = hook_env(args, 1, "observe");
    if (e == NULL) return NULL;
    bs_observe(e);
    Py_RETURN_NONE;
}

static PyObject* py_new_game(PyObject* self, PyObject* args) {
    (void)self;
    Env* e = hook_env(args, 1, "new_game");
    if (e == NULL) return NULL;
    bs_new_game(e);
    Py_RETURN_NONE;
}

static PyObject* py_seed(PyObject* self, PyObject* args) {
    (void)self;
    Env* e = hook_env(args, 2, "seed");
    if (e == NULL) return NULL;
    PyObject* s = PyTuple_GetItem(args, 1);
    if (!PyLong_Check(s)) {
        PyErr_SetString(PyExc_TypeError, "seed must be an int");
        return NULL;
    }
    unsigned long long v = PyLong_AsUnsignedLongLongMask(s);
    if (v == (unsigned long long)-1 && PyErr_Occurred()) return NULL;
    bs_seed(e, (uint64_t)v);
    Py_RETURN_NONE;
}
