import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';

let outputChannel: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext) {
    outputChannel = vscode.window.createOutputChannel("Zuntenium");
    
    // Command: Run QASM
    let runQasmCmd = vscode.commands.registerCommand('zuntenium.runQasm', async (uri: vscode.Uri) => {
        // If triggered from the command palette, there might not be a URI
        if (!uri) {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.fileName.endsWith('.qasm')) {
                uri = editor.document.uri;
            } else {
                vscode.window.showErrorMessage("Please open a .qasm file to run with Zuntenium.");
                return;
            }
        }

        const config = vscode.workspace.getConfiguration('zuntenium');
        const pythonPath = config.get<string>('pythonPath', 'python');
        const backend = config.get<string>('defaultBackend', 'statevector_simulator');
        const shots = config.get<number>('defaultShots', 1024);

        const filePath = uri.fsPath;
        const fileName = path.basename(filePath);

        outputChannel.show();
        outputChannel.appendLine(`[Running] Zuntenium Simulator on ${fileName}...`);

        const cmd = `"${pythonPath}" -m zuntenium run "${filePath}" --backend ${backend} --shots ${shots}`;
        
        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: `Zuntenium: Running ${fileName}`,
            cancellable: false
        }, async (progress) => {
            return new Promise<void>((resolve, reject) => {
                cp.exec(cmd, (error, stdout, stderr) => {
                    if (stdout) {
                        outputChannel.appendLine(stdout);
                    }
                    if (stderr) {
                        outputChannel.appendLine(stderr);
                    }
                    if (error) {
                        outputChannel.appendLine(`[Error] Execution failed: ${error.message}`);
                        vscode.window.showErrorMessage(`Zuntenium failed to run ${fileName}. See Output for details.`);
                        reject();
                    } else {
                        outputChannel.appendLine(`[Done] Simulation completed successfully.`);
                        vscode.window.showInformationMessage(`Zuntenium: Successfully ran ${fileName}`);
                        resolve();
                    }
                });
            });
        });
    });

    // Command: Show Info
    let showInfoCmd = vscode.commands.registerCommand('zuntenium.showInfo', async () => {
        const config = vscode.workspace.getConfiguration('zuntenium');
        const pythonPath = config.get<string>('pythonPath', 'python');

        outputChannel.show();
        outputChannel.appendLine(`[Info] Fetching Zuntenium system information...`);

        cp.exec(`"${pythonPath}" -m zuntenium info`, (error, stdout, stderr) => {
            if (stdout) {
                outputChannel.appendLine(stdout);
            }
            if (stderr) {
                outputChannel.appendLine(stderr);
            }
            if (error) {
                vscode.window.showErrorMessage("Failed to get Zuntenium info. Ensure it is installed in the configured Python environment.");
            }
        });
    });

    context.subscriptions.push(runQasmCmd, showInfoCmd, outputChannel);
}

export function deactivate() {
    if (outputChannel) {
        outputChannel.dispose();
    }
}
