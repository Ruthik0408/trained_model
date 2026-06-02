import { execFile, spawn } from "node:child_process";

const DEV_URL = "http://127.0.0.1:5173";
const chromeCandidates = ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"];

const vite = spawn("npx", ["vite", "--host", "127.0.0.1", "--port", "5173"], {
  stdio: "inherit",
  shell: false,
});

vite.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});

let opening = false;
let opened = false;

const isDevUrlAlreadyOpen = () =>
  new Promise((resolve) => {
    execFile("pgrep", ["-f", DEV_URL], (error, stdout) => {
      resolve(!error && stdout.trim().length > 0);
    });
  });

const tryOpenChrome = async () => {
  if (opening || opened) {
    return;
  }
  opening = true;

  try {
    const response = await fetch(DEV_URL);
    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }
  } catch {
    opening = false;
    setTimeout(tryOpenChrome, 750);
    return;
  }

  if (await isDevUrlAlreadyOpen()) {
    opened = true;
    return;
  }

  for (const command of chromeCandidates) {
    const launched = await new Promise((resolve) => {
      const browser = spawn(command, [DEV_URL], {
        detached: true,
        stdio: "ignore",
        shell: false,
      });

      browser.once("error", () => resolve(false));
      browser.once("spawn", () => {
        browser.unref();
        resolve(true);
      });
    });

    if (launched) {
      opened = true;
      return;
    }
  }

  console.warn("Could not find Chrome/Chromium on PATH. Open this URL manually:", DEV_URL);
  opened = true;
};

setTimeout(tryOpenChrome, 750);

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    vite.kill(signal);
  });
}
