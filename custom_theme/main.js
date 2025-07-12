document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('terminal-input');
    const output = document.getElementById('terminal-output');

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const command = input.value.trim();
            output.innerHTML += `<p>> ${command}</p>`;
            simulateCommand(command);
            input.value = '';
            output.scrollTop = output.scrollHeight;
        }
    });

    function simulateCommand(cmd) {
        let response = '';
        if (cmd.startsWith('protokit generate')) {
            response = 'Generating wrapper... Code preview: class CustomWrapper(gym.Wrapper) { ... }';
        } else if (cmd.startsWith('protokit tune')) {
            response = 'Tuning hyperparameters... Best LR: 0.001, Model saved.';
        } else if (cmd.startsWith('protokit debug')) {
            response = 'Debugging policy... Step 1: Action 0, Q-values [0.5, 0.3]';
        } else if (cmd.startsWith('protokit full')) {
            response = 'Running full pipeline... Wrapper generated, tuned, and debugged!';
        } else {
            response = 'Command not recognized. Try "protokit --help"';
        }
        output.innerHTML += `<p>${response}</p>`;
    }

    // Add CRT filter dynamically
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.innerHTML = `<filter id="crt-filter"><feGaussianBlur stdDeviation="0.5" /></filter>`;
    document.body.appendChild(svg);
});
