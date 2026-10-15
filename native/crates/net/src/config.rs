//! `config.json` of a run dir: the fields that decide the architecture and the inputs.
//!
//! Ported from `unified.ModesSettings` (`src/slinky/unified.py:98-129`) and `az_modes.Config`
//! (`baselines/az_modes.py:78-110`). Unknown fields (training hyperparameters) are ignored.

use crate::error::NetError;
use serde::Deserialize;

/// Rows of the egocentric window (`2*11-1`, `observations.py:118-125`).
pub const OBS_ROWS: usize = 21;
/// Columns of the egocentric window.
pub const OBS_COLS: usize = 21;
/// Observation planes (`SPEC.md` 3.1).
pub const OBS_CHANNELS: usize = 13;
/// Floats in one observation, `[21][21][13]` row-major.
pub const OBS_LEN: usize = OBS_ROWS * OBS_COLS * OBS_CHANNELS;
/// Side features per snake (`modes.NUM_FEATURES`).
pub const NUM_FEATURES: usize = 7;
/// Moves: UP, DOWN, LEFT, RIGHT.
pub const NUM_ACTIONS: usize = 4;

/// Which side features the network sees (`unified.ModesSettings.features`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureMode {
    /// All seven features.
    All,
    /// Only `turn` and `opponents_alive` (columns 4 and 6); the rest are multiplied by zero.
    NoMode,
    /// No feature layer at all.
    None,
}

/// The network-relevant part of `config.json`, validated.
#[derive(Debug, Clone, PartialEq)]
pub struct NetConfig {
    /// Output channels of each 3x3 convolution.
    pub conv_channels: Vec<usize>,
    /// Stride of each convolution (same length as `conv_channels`).
    pub conv_strides: Vec<usize>,
    /// Width of the dense hidden layer.
    pub hidden: usize,
    /// How the side features are used.
    pub features: FeatureMode,
    /// Zero the food plane when the real ruleset is constrictor (used by the observation code).
    pub hide_constrictor_food: bool,
    /// Modes the network was trained on (`standard`, `duel`, `royale`, `constrictor`).
    pub game_modes: Vec<String>,
    /// PUCT constant for the search (`az_modes.py:179`).
    pub c_puct: f32,
    /// First-play-urgency reduction for the search.
    pub fpu_reduction: f32,
    /// Value of a drawn game in the search.
    pub draw_value: f32,
}

#[derive(Deserialize)]
struct Raw {
    algorithm: Option<String>,
    #[serde(default = "default_modes")]
    game_modes: Vec<String>,
    #[serde(default = "default_features")]
    features: String,
    #[serde(default = "yes")]
    hide_constrictor_food: bool,
    #[serde(default = "default_channels")]
    conv_channels: Vec<usize>,
    #[serde(default = "default_strides")]
    conv_strides: Vec<usize>,
    #[serde(default = "default_hidden")]
    hidden: usize,
    #[serde(default = "default_dtype")]
    compute_dtype: String,
    #[serde(default = "default_c_puct")]
    c_puct: f32,
    #[serde(default = "default_fpu")]
    fpu_reduction: f32,
    #[serde(default = "default_draw")]
    draw_value: f32,
}

fn default_modes() -> Vec<String> {
    ["standard", "duel", "royale", "constrictor"]
        .map(String::from)
        .to_vec()
}
fn default_features() -> String {
    "all".into()
}
fn yes() -> bool {
    true
}
fn default_channels() -> Vec<usize> {
    vec![16, 32, 32]
}
fn default_strides() -> Vec<usize> {
    vec![2, 2, 1]
}
fn default_hidden() -> usize {
    256
}
fn default_dtype() -> String {
    "float32".into()
}
fn default_c_puct() -> f32 {
    0.5
}
fn default_fpu() -> f32 {
    0.2
}
fn default_draw() -> f32 {
    -0.5
}

impl NetConfig {
    /// Parse and validate the text of a `config.json`.
    pub fn from_json(text: &str) -> Result<NetConfig, NetError> {
        let r: Raw = serde_json::from_str(text).map_err(|e| NetError::Json(e.to_string()))?;
        if let Some(a) = &r.algorithm
            && a != "az_modes"
        {
            return Err(NetError::Unsupported(format!(
                "algorithm {a:?}, expected \"az_modes\""
            )));
        }
        if r.compute_dtype != "float32" {
            return Err(NetError::Unsupported(format!(
                "compute_dtype {:?} (only float32 is implemented)",
                r.compute_dtype
            )));
        }
        let features = match r.features.as_str() {
            "all" => FeatureMode::All,
            "no_mode" => FeatureMode::NoMode,
            "none" => FeatureMode::None,
            f => return Err(NetError::Unsupported(format!("features {f:?}"))),
        };
        if r.conv_channels.len() != r.conv_strides.len() {
            return Err(NetError::Unsupported(
                "conv_channels and conv_strides differ in length".into(),
            ));
        }
        if r.conv_channels.is_empty()
            || r.conv_channels.contains(&0)
            || r.conv_strides.contains(&0)
            || r.hidden == 0
        {
            return Err(NetError::Unsupported(
                "empty or zero-sized layer in config".into(),
            ));
        }
        if r.conv_channels.len() > 16
            || r.conv_channels
                .iter()
                .chain(&r.conv_strides)
                .any(|&c| c > 4096)
            || r.hidden > 1 << 16
        {
            return Err(NetError::Unsupported(
                "implausibly large layer sizes in config".into(),
            ));
        }
        Ok(NetConfig {
            conv_channels: r.conv_channels,
            conv_strides: r.conv_strides,
            hidden: r.hidden,
            features,
            hide_constrictor_food: r.hide_constrictor_food,
            game_modes: r.game_modes,
            c_puct: r.c_puct,
            fpu_reduction: r.fpu_reduction,
            draw_value: r.draw_value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_errors() {
        let c = NetConfig::from_json(r#"{"algorithm":"az_modes","unknown":1}"#).unwrap();
        assert_eq!(c.conv_channels, vec![16, 32, 32]);
        assert!(NetConfig::from_json(r#"{"compute_dtype":"bfloat16"}"#).is_err());
        assert!(NetConfig::from_json(r#"{"algorithm":"ppo"}"#).is_err());
        assert!(NetConfig::from_json(r#"{"features":"x"}"#).is_err());
        assert!(NetConfig::from_json(r#"{"conv_strides":[1]}"#).is_err());
        assert!(NetConfig::from_json("nonsense").is_err());
    }
}
