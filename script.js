document.addEventListener('DOMContentLoaded', function() {
    // Add a smooth scroll effect
    const scrollBtn = document.querySelector('.scroll-btn');
    scrollBtn.addEventListener('click', function() {
        window.scrollTo({ top: 500, behavior: 'smooth' });
    });
});