use chess_common::{Move, Score};
use std::fmt;

/// A command sent from the GUI to the engine.
#[derive(Debug, Clone, PartialEq)]
pub enum UciCommand {
    Uci,
    Debug(bool),
    IsReady,
    SetOption { name: String, value: Option<String> },
    Register,
    UciNewGame,
    Position { fen: Option<String>, moves: Vec<String> },
    Go(GoParams),
    Stop,
    PonderHit,
    Quit,
}

/// Parameters for the `go` command.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GoParams {
    pub searchmoves: Vec<String>,
    pub ponder: bool,
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub movestogo: Option<u32>,
    pub depth: Option<u8>,
    pub nodes: Option<u64>,
    pub mate: Option<u32>,
    pub movetime: Option<u64>,
    pub infinite: bool,
}

/// A response sent from the engine to the GUI.
#[derive(Debug, Clone)]
pub enum UciResponse {
    Id { name: String, author: String },
    UciOk,
    ReadyOk,
    BestMove { best: Move, ponder: Option<Move> },
    Info(UciInfo),
    Option(UciOptionDef),
}

/// Info fields for a UCI `info` line.
#[derive(Debug, Clone, Default)]
pub struct UciInfo {
    pub depth: Option<u8>,
    pub seldepth: Option<u8>,
    /// MultiPV line number (1-based). `None` in single-PV mode.
    pub multipv: Option<usize>,
    pub time: Option<u64>,
    pub nodes: Option<u64>,
    pub pv: Vec<Move>,
    pub score: Option<Score>,
    /// WDL in millipawns (win, draw, loss), each 0–1000, summing to 1000.
    /// Only set when UCI_ShowWDL is enabled.
    pub wdl: Option<(u32, u32, u32)>,
    pub hashfull: Option<u16>,
    pub nps: Option<u64>,
    pub string: Option<String>,
}

/// Definition of a UCI option.
#[derive(Debug, Clone)]
pub struct UciOptionDef {
    pub name: String,
    pub opt_type: UciOptionType,
}

#[derive(Debug, Clone)]
pub enum UciOptionType {
    Check { default: bool },
    Spin { default: i64, min: i64, max: i64 },
    Combo { default: String, options: Vec<String> },
    Button,
    String { default: String },
}

impl UciCommand {
    /// Parse a UCI command from a line of text.
    pub fn parse(line: &str) -> Option<Self> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }

        let mut tokens = line.split_whitespace();
        let cmd = tokens.next()?;

        match cmd {
            "uci" => Some(UciCommand::Uci),
            "debug" => {
                let on = tokens.next().unwrap_or("off") == "on";
                Some(UciCommand::Debug(on))
            }
            "isready" => Some(UciCommand::IsReady),
            "setoption" => parse_setoption(&mut tokens),
            "register" => Some(UciCommand::Register),
            "ucinewgame" => Some(UciCommand::UciNewGame),
            "position" => parse_position(&mut tokens),
            "go" => parse_go(&mut tokens),
            "stop" => Some(UciCommand::Stop),
            "ponderhit" => Some(UciCommand::PonderHit),
            "quit" => Some(UciCommand::Quit),
            _ => {
                log::debug!("Unknown UCI command: {}", cmd);
                None
            }
        }
    }
}

fn parse_setoption<'a>(tokens: &mut impl Iterator<Item = &'a str>) -> Option<UciCommand> {
    // setoption name <name> [value <value>]
    // Skip "name" keyword
    let first = tokens.next()?;
    if first != "name" {
        return None;
    }

    let mut name_parts = Vec::new();
    let mut value_parts = Vec::new();
    let mut in_value = false;

    for token in tokens {
        if token == "value" && !in_value {
            in_value = true;
            continue;
        }
        if in_value {
            value_parts.push(token);
        } else {
            name_parts.push(token);
        }
    }

    let name = name_parts.join(" ");
    let value = if in_value {
        Some(value_parts.join(" "))
    } else {
        None
    };

    Some(UciCommand::SetOption { name, value })
}

fn parse_position<'a>(tokens: &mut impl Iterator<Item = &'a str>) -> Option<UciCommand> {
    let pos_type = tokens.next()?;

    let mut fen = None;
    let mut moves = Vec::new();
    let remaining: Vec<&str> = tokens.collect();

    match pos_type {
        "startpos" => {
            // Find "moves" keyword if present
            if let Some(moves_idx) = remaining.iter().position(|&t| t == "moves") {
                moves = remaining[moves_idx + 1..].iter().map(|s| s.to_string()).collect();
            }
        }
        "fen" => {
            // Collect FEN parts until we hit "moves" or run out
            if let Some(moves_idx) = remaining.iter().position(|&t| t == "moves") {
                fen = Some(remaining[..moves_idx].join(" "));
                moves = remaining[moves_idx + 1..].iter().map(|s| s.to_string()).collect();
            } else {
                fen = Some(remaining.join(" "));
            }
        }
        _ => return None,
    }

    Some(UciCommand::Position { fen, moves })
}

fn parse_go<'a>(tokens: &mut impl Iterator<Item = &'a str>) -> Option<UciCommand> {
    let mut params = GoParams::default();
    let tokens: Vec<&str> = tokens.collect();
    let mut i = 0;

    while i < tokens.len() {
        match tokens[i] {
            "searchmoves" => {
                i += 1;
                // Collect all subsequent tokens that look like moves (until next keyword)
                while i < tokens.len() && !is_go_keyword(tokens[i]) {
                    params.searchmoves.push(tokens[i].to_string());
                    i += 1;
                }
                continue;
            }
            "ponder" => params.ponder = true,
            "wtime" => {
                i += 1;
                if i < tokens.len() {
                    params.wtime = tokens[i].parse().ok();
                }
            }
            "btime" => {
                i += 1;
                if i < tokens.len() {
                    params.btime = tokens[i].parse().ok();
                }
            }
            "winc" => {
                i += 1;
                if i < tokens.len() {
                    params.winc = tokens[i].parse().ok();
                }
            }
            "binc" => {
                i += 1;
                if i < tokens.len() {
                    params.binc = tokens[i].parse().ok();
                }
            }
            "movestogo" => {
                i += 1;
                if i < tokens.len() {
                    params.movestogo = tokens[i].parse().ok();
                }
            }
            "depth" => {
                i += 1;
                if i < tokens.len() {
                    params.depth = tokens[i].parse().ok();
                }
            }
            "nodes" => {
                i += 1;
                if i < tokens.len() {
                    params.nodes = tokens[i].parse().ok();
                }
            }
            "mate" => {
                i += 1;
                if i < tokens.len() {
                    params.mate = tokens[i].parse().ok();
                }
            }
            "movetime" => {
                i += 1;
                if i < tokens.len() {
                    params.movetime = tokens[i].parse().ok();
                }
            }
            "infinite" => params.infinite = true,
            _ => {}
        }
        i += 1;
    }

    Some(UciCommand::Go(params))
}

fn is_go_keyword(s: &str) -> bool {
    matches!(
        s,
        "searchmoves"
            | "ponder"
            | "wtime"
            | "btime"
            | "winc"
            | "binc"
            | "movestogo"
            | "depth"
            | "nodes"
            | "mate"
            | "movetime"
            | "infinite"
    )
}

impl fmt::Display for UciResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UciResponse::Id { name, author } => {
                write!(f, "id name {name}\nid author {author}")
            }
            UciResponse::UciOk => write!(f, "uciok"),
            UciResponse::ReadyOk => write!(f, "readyok"),
            UciResponse::BestMove { best, ponder } => {
                write!(f, "bestmove {}", best.to_uci())?;
                if let Some(p) = ponder {
                    write!(f, " ponder {}", p.to_uci())?;
                }
                Ok(())
            }
            UciResponse::Info(info) => {
                write!(f, "info")?;
                if let Some(depth) = info.depth {
                    write!(f, " depth {depth}")?;
                }
                if let Some(seldepth) = info.seldepth {
                    write!(f, " seldepth {seldepth}")?;
                }
                if let Some(multipv) = info.multipv {
                    write!(f, " multipv {multipv}")?;
                }
                if let Some(score) = info.score {
                    write!(f, " score {score}")?;
                }
                if let Some((w, d, l)) = info.wdl {
                    write!(f, " wdl {w} {d} {l}")?;
                }
                if let Some(nodes) = info.nodes {
                    write!(f, " nodes {nodes}")?;
                }
                if let Some(nps) = info.nps {
                    write!(f, " nps {nps}")?;
                }
                if let Some(hashfull) = info.hashfull {
                    write!(f, " hashfull {hashfull}")?;
                }
                if let Some(time) = info.time {
                    write!(f, " time {time}")?;
                }
                if !info.pv.is_empty() {
                    write!(f, " pv")?;
                    for m in &info.pv {
                        write!(f, " {}", m.to_uci())?;
                    }
                }
                if let Some(ref s) = info.string {
                    write!(f, " string {s}")?;
                }
                Ok(())
            }
            UciResponse::Option(opt) => {
                write!(f, "option name {} type ", opt.name)?;
                match &opt.opt_type {
                    UciOptionType::Check { default } => {
                        write!(f, "check default {default}")
                    }
                    UciOptionType::Spin { default, min, max } => {
                        write!(f, "spin default {default} min {min} max {max}")
                    }
                    UciOptionType::Combo { default, options } => {
                        write!(f, "combo default {default}")?;
                        for opt in options {
                            write!(f, " var {opt}")?;
                        }
                        Ok(())
                    }
                    UciOptionType::Button => write!(f, "button"),
                    UciOptionType::String { default } => {
                        write!(f, "string default {default}")
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_uci() {
        assert_eq!(UciCommand::parse("uci"), Some(UciCommand::Uci));
    }

    #[test]
    fn test_parse_isready() {
        assert_eq!(UciCommand::parse("isready"), Some(UciCommand::IsReady));
    }

    #[test]
    fn test_parse_quit() {
        assert_eq!(UciCommand::parse("quit"), Some(UciCommand::Quit));
    }

    #[test]
    fn test_parse_stop() {
        assert_eq!(UciCommand::parse("stop"), Some(UciCommand::Stop));
    }

    #[test]
    fn test_parse_ucinewgame() {
        assert_eq!(UciCommand::parse("ucinewgame"), Some(UciCommand::UciNewGame));
    }

    #[test]
    fn test_parse_position_startpos() {
        assert_eq!(
            UciCommand::parse("position startpos"),
            Some(UciCommand::Position {
                fen: None,
                moves: vec![]
            })
        );
    }

    #[test]
    fn test_parse_position_startpos_moves() {
        assert_eq!(
            UciCommand::parse("position startpos moves e2e4 e7e5"),
            Some(UciCommand::Position {
                fen: None,
                moves: vec!["e2e4".to_string(), "e7e5".to_string()]
            })
        );
    }

    #[test]
    fn test_parse_position_fen() {
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let cmd = UciCommand::parse(&format!("position fen {fen}"));
        assert_eq!(
            cmd,
            Some(UciCommand::Position {
                fen: Some(fen.to_string()),
                moves: vec![]
            })
        );
    }

    #[test]
    fn test_parse_position_fen_with_moves() {
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let cmd = UciCommand::parse(&format!("position fen {fen} moves e7e5"));
        assert_eq!(
            cmd,
            Some(UciCommand::Position {
                fen: Some(fen.to_string()),
                moves: vec!["e7e5".to_string()]
            })
        );
    }

    #[test]
    fn test_parse_go_depth() {
        let cmd = UciCommand::parse("go depth 10");
        assert_eq!(
            cmd,
            Some(UciCommand::Go(GoParams {
                depth: Some(10),
                ..GoParams::default()
            }))
        );
    }

    #[test]
    fn test_parse_go_infinite() {
        let cmd = UciCommand::parse("go infinite");
        assert_eq!(
            cmd,
            Some(UciCommand::Go(GoParams {
                infinite: true,
                ..GoParams::default()
            }))
        );
    }

    #[test]
    fn test_parse_go_time_controls() {
        let cmd = UciCommand::parse("go wtime 300000 btime 300000 winc 2000 binc 2000");
        assert_eq!(
            cmd,
            Some(UciCommand::Go(GoParams {
                wtime: Some(300000),
                btime: Some(300000),
                winc: Some(2000),
                binc: Some(2000),
                ..GoParams::default()
            }))
        );
    }

    #[test]
    fn test_parse_setoption() {
        let cmd = UciCommand::parse("setoption name Hash value 128");
        assert_eq!(
            cmd,
            Some(UciCommand::SetOption {
                name: "Hash".to_string(),
                value: Some("128".to_string())
            })
        );
    }

    #[test]
    fn test_parse_setoption_no_value() {
        let cmd = UciCommand::parse("setoption name Clear Hash");
        assert_eq!(
            cmd,
            Some(UciCommand::SetOption {
                name: "Clear Hash".to_string(),
                value: None
            })
        );
    }

    #[test]
    fn test_parse_debug() {
        assert_eq!(UciCommand::parse("debug on"), Some(UciCommand::Debug(true)));
        assert_eq!(
            UciCommand::parse("debug off"),
            Some(UciCommand::Debug(false))
        );
    }

    #[test]
    fn test_parse_empty() {
        assert_eq!(UciCommand::parse(""), None);
        assert_eq!(UciCommand::parse("   "), None);
    }

    #[test]
    fn test_format_bestmove() {
        let m = Move::from_uci("e2e4").unwrap();
        let resp = UciResponse::BestMove {
            best: m,
            ponder: None,
        };
        assert_eq!(resp.to_string(), "bestmove e2e4");
    }

    #[test]
    fn test_format_bestmove_with_ponder() {
        let best = Move::from_uci("e2e4").unwrap();
        let ponder = Move::from_uci("e7e5").unwrap();
        let resp = UciResponse::BestMove {
            best,
            ponder: Some(ponder),
        };
        assert_eq!(resp.to_string(), "bestmove e2e4 ponder e7e5");
    }

    #[test]
    fn test_format_info() {
        let info = UciInfo {
            depth: Some(5),
            seldepth: Some(8),
            score: Some(Score(50)),
            nodes: Some(12345),
            time: Some(100),
            nps: Some(123450),
            pv: vec![
                Move::from_uci("e2e4").unwrap(),
                Move::from_uci("e7e5").unwrap(),
            ],
            ..UciInfo::default()
        };
        let resp = UciResponse::Info(info);
        assert_eq!(
            resp.to_string(),
            "info depth 5 seldepth 8 score cp 50 nodes 12345 nps 123450 time 100 pv e2e4 e7e5"
        );
    }

    #[test]
    fn test_format_info_mate_score() {
        let info = UciInfo {
            depth: Some(10),
            score: Some(Score(29999)),
            ..UciInfo::default()
        };
        let resp = UciResponse::Info(info);
        assert_eq!(resp.to_string(), "info depth 10 score mate 1");
    }
}
