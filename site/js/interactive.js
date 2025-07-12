document.addEventListener('DOMContentLoaded', () => {
  // Playful toggle for code blocks (TE-inspired modularity)
  document.querySelectorAll('.md-typeset__scrollwrap').forEach(el => {
    el.addEventListener('click', () => {
      el.style.backgroundColor = '#ff69b4';  // Hot pink flash
      setTimeout(() => el.style.backgroundColor = '', 300);
    });
  });
  // Simple code playground (run dummy RL sim)
  const playground = document.createElement('button');
  playground.textContent = 'Run Demo';
  playground.onclick = () => alert('Simulating RL step: Action chosen!');
  document.body.appendChild(playground);
});