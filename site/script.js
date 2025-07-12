// Command Execution and Detailed Animations
function runCommand(type) {
    const output = document.getElementById('terminal-output');
    output.innerHTML = `<p>Executing ${type}...</p>`;
    
    const steps = {
        generate: [
            {name: 'Parse Mods', desc: 'Input string is split into key-value pairs (e.g., "scale_rewards=0.5" becomes key="scale_rewards", value=0.5). This allows flexible customization without hardcoding.', animation: 'break-chains'},
            {name: 'Build Template', desc: 'Constructs the CustomWrapper class by inserting mod logic into a string template. For example, adds "reward *= 0.5" to the step method for reward scaling.', animation: 'assemble-parts'},
            {name: 'Output File', desc: 'Writes the complete code to a .py file, ready for import in your RL script. Final output is a functional wrapper class.', animation: 'file-pop'}
        ],
        tune: [
            {name: 'Create Grid', desc: 'Uses itertools to generate all parameter combinations (e.g., different learning rates). This ensures exhaustive testing within the grid.', animation: 'matrix-build'},
            {name: 'Run Parallel Trials', desc: 'Launches training in a multiprocessing Pool for speed, running limited episodes per trial to keep it lightweight.', animation: 'racing-cars'},
            {name: 'Rank & Save', desc: 'Aggregates results in Pandas, ranks by average reward, and saves the best model as tuned_model.pth for later use.', animation: 'podium-rank'}
        ],
        debug: [
            {name: 'Load Model', desc: 'Uses torch.load to import the trained model from file, preparing it for evaluation in eval mode.', animation: 'wire-connect'},
            {name: 'Simulate Steps', desc: 'Loops through env.step, computing actions and Q-values with the model, handling terminations.', animation: 'walking-figure'},
            {name: 'Display Values', desc: 'Builds a Rich table showing state, action, Q-values, and confidence, highlighting anomalies like low confidence.', animation: 'bubble-pop'}
        ],
        full: [
            {name: 'Generate Wrapper', desc: 'Calls generate to create and modify the environment wrapper.', animation: 'link-chain'},
            {name: 'Tune Params', desc: 'Runs tuning on the wrapped env to optimize hyperparameters.', animation: 'gear-turn'},
            {name: 'Debug Policy', desc: 'Debugs the tuned model, chaining the full workflow for complete results.', animation: 'final-explosion'}
        ]
    }[type];

    steps.forEach((step, index) => {
        setTimeout(() => {
            output.innerHTML += `<p class="animation-step">${step.name}: ${step.desc}</p>`;
            animateComponent(step.animation);
        }, index * 2000); // Slower pacing for reading
    });
}

// Animation Function for Components
function animateComponent(type) {
    const container = document.getElementById('animation-container');
    const anim = document.createElement('div');
    anim.className = `animation ${type}`;
    container.appendChild(anim);
    setTimeout(() => anim.remove(), 1500);
}

// Manual Input
document.getElementById('terminal-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        const type = e.target.value.split(' ')[1] || 'generate';
        runCommand(type);
        e.target.value = '';
    }
});

// Flashing Toggle
function toggleFlashing() {
    const enable = document.getElementById('flash-enable').checked;
    document.body.classList.toggle('flashing', enable);
}

// Music Player
const audio = document.getElementById('audio-player');
audio.volume = 0.5;
audio.play(); // Auto-start

function togglePlay() {
    if (audio.paused) {
        audio.play();
        document.querySelectorAll('.reel').forEach(reel => reel.style.animationPlayState = 'running');
    } else {
        audio.pause();
        document.querySelectorAll('.reel').forEach(reel => reel.style.animationPlayState = 'paused');
    }
}

function setVolume(value) {
    audio.volume = parseFloat(value);
}

// Song Search with Jamendo API
function searchSong() {
    const query = document.getElementById('song-search').value;
    const clientId = '44cdcf9f';
    const clientSecret = '5e0155f0852ea1b97ca9fd49b4e84a0e';
    fetch(`https://api.jamendo.com/v3.0/tracks/?client_id=${clientId}&format=jsonpretty&limit=1&search=${query}&access_token=${clientSecret}`)
        .then(response => response.json())
        .then(data => {
            if (data.results && data.results.length > 0) {
                audio.src = data.results[0].audio;
                audio.play();
                document.querySelector('.player-display').textContent = data.results[0].name;
            } else {
                alert('No tracks found; playing default.');
                audio.src = 'on_sight.mp3';
                audio.play();
                document.querySelector('.player-display').textContent = 'On Sight - Kanye West';
            }
        })
        .catch(error => {
            console.error('API error:', error);
            alert('API issue; playing local default.');
            audio.src = 'on_sight.mp3';
            audio.play();
        });
}
