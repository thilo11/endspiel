package dev.thilo.endspiel.oex;

import android.app.Activity;
import android.os.Bundle;

/**
 * The OEX protocol requires a launchable activity to carry the engine meta-data,
 * but this plugin has no UI of its own — it just exposes the bundled UCI binary
 * to a chess GUI. So the activity finishes immediately if a user opens it.
 */
public class MainActivity extends Activity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        finish();
    }
}
