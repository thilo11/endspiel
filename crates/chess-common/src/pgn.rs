use crate::types::GameResult;

/// A parsed PGN game.
#[derive(Debug, Clone)]
pub struct PgnGame {
    pub headers: Vec<(String, String)>,
    pub moves: Vec<String>, // SAN move strings
    pub result: Option<GameResult>,
}

impl PgnGame {
    pub fn new() -> Self {
        Self {
            headers: Vec::new(),
            moves: Vec::new(),
            result: None,
        }
    }

    /// Parse a simple PGN string.
    pub fn from_pgn(input: &str) -> Result<Self, String> {
        let mut game = PgnGame::new();
        let mut in_moves = false;
        let mut move_text = String::new();

        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() {
                if !game.headers.is_empty() {
                    in_moves = true;
                }
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') && !in_moves {
                // Parse header
                let inner = &line[1..line.len() - 1];
                if let Some(space_pos) = inner.find(' ') {
                    let key = inner[..space_pos].to_string();
                    let val = inner[space_pos + 1..].trim().trim_matches('"').to_string();
                    game.headers.push((key, val));
                }
            } else {
                in_moves = true;
                move_text.push(' ');
                move_text.push_str(line);
            }
        }

        // Parse move text
        let tokens: Vec<&str> = move_text.split_whitespace().collect();
        for token in &tokens {
            let token = *token;
            // Skip move numbers (e.g., "1.", "1...")
            if token.ends_with('.') || token.contains("...") {
                continue;
            }
            // Check for result tokens
            match token {
                "1-0" => {
                    game.result = Some(GameResult::WhiteWins);
                    continue;
                }
                "0-1" => {
                    game.result = Some(GameResult::BlackWins);
                    continue;
                }
                "1/2-1/2" => {
                    game.result = Some(GameResult::Draw(
                        crate::types::DrawReason::Agreement,
                    ));
                    continue;
                }
                "*" => {
                    game.result = Some(GameResult::Ongoing);
                    continue;
                }
                _ => {}
            }
            // Skip annotations
            if token.starts_with('{') || token.starts_with('(') || token.starts_with('$') {
                continue;
            }
            game.moves.push(token.to_string());
        }

        Ok(game)
    }

    /// Convert a game to PGN format.
    pub fn to_pgn(&self) -> String {
        let mut out = String::new();
        for (key, val) in &self.headers {
            out.push_str(&format!("[{key} \"{val}\"]\n"));
        }
        out.push('\n');

        for (i, m) in self.moves.iter().enumerate() {
            if i % 2 == 0 {
                out.push_str(&format!("{}. ", i / 2 + 1));
            }
            out.push_str(m);
            out.push(' ');
        }

        match self.result {
            Some(GameResult::WhiteWins) => out.push_str("1-0"),
            Some(GameResult::BlackWins) => out.push_str("0-1"),
            Some(GameResult::Draw(_)) => out.push_str("1/2-1/2"),
            _ => out.push('*'),
        }

        out.push('\n');
        out
    }

    pub fn get_header(&self, key: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

impl Default for PgnGame {
    fn default() -> Self {
        Self::new()
    }
}
