package dev.thilo.endspiel.oex;

import com.kalab.chess.enginesupport.ChessEngineProvider;

/**
 * Serves the bundled UCI binary to a requesting chess GUI. All logic lives in
 * the Apache-2.0 {@link ChessEngineProvider} base class; this subclass exists
 * only so the manifest can name a provider in this app's package.
 */
public class EngineProvider extends ChessEngineProvider {
}
