// docs/script.js

// --- CRT Terminal Command Logic and Animations ---

function runCommand(type) {
    const output = document.getElementById('terminal-output');
    output.innerHTML = `<p>Executing ${type}...</p>`;
    const steps = {
        generate: [
            {name: 'Parse Mods', desc: 'Splits input like "scale_rewards=0.5" into key-value. This allows flexible customization without hardcoding.', animation: 'break-chains'},
            {name: 'Build Template', desc: 'Constructs the CustomWrapper class by inserting mod logic into a string template. For example, adds "reward *= 0.5" to the step method for reward scaling.', animation: 'assemble-parts'},
            {name: 'Output File', desc: 'Writes the complete code to a .py file, ready for import in your RL script. Final output is a functional wrapper class.', animation: 'file-pop'}
        ],
        tune: [
            {name: 'Create Grid', desc: 'Uses itertools to generate all parameter combinations (e.g., different learning rates). Ensures exhaustive testing.', animation: 'matrix-build'},
            {name: 'Run Parallel Trials', desc: 'Launches training in a multiprocessing Pool for speed, running limited episodes per trial.', animation: 'racing-cars'},
            {name: 'Rank & Save', desc: 'Aggregates results in Pandas, ranks by average reward, and saves the best model as tuned_model.pth.', animation: 'podium-rank'}
        ],
        debug: [
            {name: 'Load Model', desc: 'Uses torch.load to import the trained model from file, preparing it for evaluation in eval mode.', animation: 'wire-connect'},
            {name: 'Simulate Steps', desc: 'Loops through env.step, computing actions and Q-values with the model, handling terminations.', animation: 'walking-figure'},
            {name: 'Display Values', desc: 'Builds a Rich table showing state, action, Q-values, and confidence, highlighting anomalies.', animation: 'bubble-pop'}
        ],
        full: [
            {name: 'Generate Wrapper', desc: 'Calls generate to create and modify the environment wrapper.', animation: 'link-chain'},
            {name: 'Tune Params', desc: 'Runs tuning on the wrapped env to optimize hyperparameters.', animation: 'gear-turn'},
            {name: 'Debug Policy', desc: 'Debugs the tuned model, chaining the full workflow.', animation: 'final-explosion'}
        ]
    }[type];
    steps.forEach((step, index) => {
        setTimeout(() => {
            output.innerHTML += `<p class="animation-step"><b>${step.name}:</b> ${step.desc}</p>`;
            animateComponent(step.animation);
        }, index * 1500);
    });
}

function animateComponent(type) {
    const container = document.getElementById('animation-container');
    const anim = document.createElement('div');
    anim.className = `animation ${type}`;
    container.appendChild(anim);
    setTimeout(() => anim.remove(), 1500);
}

document.getElementById('terminal-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        const type = e.target.value.split(' ')[1] || 'generate';
        runCommand(type);
        e.target.value = '';
    }
});

// --- Flashing Effect (Whole Window) ---

function toggleFlashing() {
    const enable = document.getElementById('flash-enable').checked;
    if (enable) {
        document.body.classList.add('flashing');
        document.querySelector('.crt-terminal').classList.add('flashing');
    } else {
        document.body.classList.remove('flashing');
        document.querySelector('.crt-terminal').classList.remove('flashing');
    }
}

// --- Music Player Logic ---

const musicFolder = "music/";
const tracks = [
    {file: "on_sight.mp3", title: "On Sight - Kanye West"},
    // Add more tracks here, e.g.:
    // {file: "track2.mp3", title: "Track 2 - Artist"},
    // {file: "track3.mp3", title: "Track 3 - Artist"},
];
let currentTrack = 0;
let isShuffling = false;

const audio = document.getElementById('audio-player');
const playerDisplay = document.getElementById('player-display');

function loadTrack(idx) {
    currentTrack = idx;
    audio.src = musicFolder + tracks[currentTrack].file;
    playerDisplay.textContent = tracks[currentTrack].title;
    playMusic();
}

function playMusic() {
    audio.play();
    document.querySelectorAll('.reel').forEach(reel => reel.style.animationPlayState = 'running');
}

function pauseMusic() {
    audio.pause();
    document.querySelectorAll('.reel').forEach(reel => reel.style.animationPlayState = 'paused');
}

function togglePlay() {
    if (audio.paused) {
        playMusic();
    } else {
        pauseMusic();
    }
}

function setVolume(value) {
    audio.volume = parseFloat(value);
}

function nextTrack() {
    if (isShuffling) {
        let next = Math.floor(Math.random() * tracks.length);
        while (next === currentTrack && tracks.length > 1) {
            next = Math.floor(Math.random() * tracks.length);
        }
        loadTrack(next);
    } else {
        loadTrack((currentTrack + 1) % tracks.length);
    }
}

function previousTrack() {
    loadTrack((currentTrack - 1 + tracks.length) % tracks.length);
}

function shuffleMusic() {
    isShuffling = !isShuffling;
    document.querySelector('.shuffle-btn').classList.toggle('active', isShuffling);
    if (isShuffling) nextTrack();
}

audio.addEventListener('ended', () => {
    nextTrack();
});

// Auto-play on load and start reels
window.addEventListener('DOMContentLoaded', () => {
    loadTrack(0);
    playMusic();
});

// Clicking anywhere starts music if browser blocks autoplay
document.body.addEventListener('click', () => {
    if (audio.paused) playMusic();
    document.querySelectorAll('.reel').forEach(reel => reel.style.animationPlayState = 'running');
}, { once: true });
