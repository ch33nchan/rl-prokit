// Music Player
const musicFolder = 'music/';
const playlist = [
  'on_sight.mp3',
  // Add more mp3 filenames here, e.g. 'track2.mp3', 'track3.mp3'
];
let currentTrack = 0;
const audio = document.getElementById('audio-player');
const playerDisplay = document.getElementById('player-display');
const reels = document.querySelectorAll('.reel');

function loadTrack(idx) {
  if (idx < 0) idx = playlist.length - 1;
  if (idx >= playlist.length) idx = 0;
  currentTrack = idx;
  audio.src = musicFolder + playlist[currentTrack];
  playerDisplay.textContent = playlist[currentTrack].replace('.mp3', '').replace(/_/g, ' ');
  audio.play();
  reels.forEach(r => r.style.animationPlayState = 'running');
}

function togglePlay() {
  if (audio.paused) {
    audio.play();
    reels.forEach(r => r.style.animationPlayState = 'running');
  } else {
    audio.pause();
    reels.forEach(r => r.style.animationPlayState = 'paused');
  }
}
function setVolume(value) {
  audio.volume = parseFloat(value);
}
function nextTrack() {
  loadTrack(currentTrack + 1);
}
function previousTrack() {
  loadTrack(currentTrack - 1);
}
function shuffleMusic() {
  let idx = Math.floor(Math.random() * playlist.length);
  while (idx === currentTrack && playlist.length > 1) {
    idx = Math.floor(Math.random() * playlist.length);
  }
  loadTrack(idx);
}
audio.addEventListener('ended', nextTrack);

// Auto-play on first interaction (browser restriction workaround)
document.body.addEventListener('click', () => {
  if (audio.paused) {
    loadTrack(currentTrack);
    audio.play();
    reels.forEach(r => r.style.animationPlayState = 'running');
  }
}, { once: true });

// Load initial track
window.addEventListener('DOMContentLoaded', () => {
  loadTrack(currentTrack);
  setVolume(document.getElementById('volume-slider').value);
});

// Flashing toggle
function toggleFlashing() {
  const enabled = document.getElementById('flash-enable').checked;
  document.body.classList.toggle('flashing', enabled);
  document.querySelector('.trippy-bg').classList.toggle('flashing', enabled);
}

// CRT Terminal Command Logic
function runCommand(type) {
  const output = document.getElementById('terminal-output');
  output.innerHTML = `<p>Executing <b>${type}</b>...</p>`;
  const steps = {
    generate: [
      {name: 'Parse Mods', desc: 'Splits input like "scale_rewards=0.5" into key-value pairs.'},
      {name: 'Build Template', desc: 'Constructs the CustomWrapper class with mod logic.'},
      {name: 'Output File', desc: 'Writes the code to a .py file for import.'}
    ],
    tune: [
      {name: 'Create Grid', desc: 'Generates parameter combinations.'},
      {name: 'Run Parallel Trials', desc: 'Runs training trials in parallel.'},
      {name: 'Rank & Save', desc: 'Ranks results and saves the best model.'}
    ],
    debug: [
      {name: 'Load Model', desc: 'Loads model from file.'},
      {name: 'Simulate Steps', desc: 'Simulates environment steps.'},
      {name: 'Display Values', desc: 'Shows Q-values and confidence.'}
    ],
    full: [
      {name: 'Generate Wrapper', desc: 'Generates environment wrapper.'},
      {name: 'Tune Params', desc: 'Tunes hyperparameters.'},
      {name: 'Debug Policy', desc: 'Debugs the tuned model.'}
    ]
  }[type];

  steps.forEach((step, index) => {
    setTimeout(() => {
      output.innerHTML += `<p class="animation-step"><b>${step.name}:</b> ${step.desc}</p>`;
      animateComponent();
    }, index * 1500);
  });
}

function animateComponent() {
  const container = document.getElementById('animation-container');
  const particle = document.createElement('div');
  particle.className = 'particle';
  particle.style.left = `${Math.random() * 90 + 5}%`;
  particle.style.top = `${Math.random() * 80 + 10}%`;
  container.appendChild(particle);
  setTimeout(() => particle.remove(), 1500);
}

document.getElementById('terminal-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const type = e.target.value.split(' ')[1] || 'generate';
    runCommand(type);
    e.target.value = '';
  }
});
