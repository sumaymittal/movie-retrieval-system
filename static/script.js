document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm');
    const searchInput = document.getElementById('searchInput');
    const submitBtnText = document.querySelector('#submitBtn span');
    const loader = document.querySelector('.loader');
    
    const resultsArea = document.getElementById('resultsArea');
    const ragAnswerSection = document.getElementById('ragAnswerSection');
    const ragAnswerContent = document.getElementById('ragAnswerContent');
    const ragSourcesList = document.getElementById('ragSourcesList');
    const standardResultsSection = document.getElementById('standardResultsSection');
    
    const tabBtns = document.querySelectorAll('.tab-btn');
    let currentMode = 'search'; // 'search' or 'rag'

    // Handle tab switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;
            
            if (currentMode === 'rag') {
                searchInput.placeholder = "Ask a complex question about movies...";
            } else {
                searchInput.placeholder = "Search for a movie, plot, or ask a question...";
            }
        });
    });

    // Form submission
    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = searchInput.value.trim();
        if (!query) return;

        // UI Loading state
        submitBtnText.classList.add('hidden');
        loader.classList.remove('hidden');
        resultsArea.classList.add('hidden');
        ragAnswerSection.classList.add('hidden');
        standardResultsSection.classList.add('hidden');
        
        try {
            if (currentMode === 'search') {
                await performStandardSearch(query);
            } else {
                await performRAGSearch(query);
            }
            resultsArea.classList.remove('hidden');
        } catch (error) {
            console.error("Error fetching results:", error);
            alert("Failed to fetch results. Please try again later.");
        } finally {
            submitBtnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });

    async function performStandardSearch(query) {
        const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        
        standardResultsSection.innerHTML = '';
        
        if (data.results.length === 0) {
            standardResultsSection.innerHTML = '<div class="glass-panel result-card"><p>No movies found matching your query.</p></div>';
        } else {
            data.results.forEach(movie => {
                const card = document.createElement('div');
                card.className = 'glass-panel result-card';
                card.innerHTML = `
                    <div class="result-header">
                        <h3 class="result-title">${movie.title}</h3>
                        <span class="result-score">${movie.score}% Match</span>
                    </div>
                    <p class="result-desc">${movie.description}</p>
                `;
                standardResultsSection.appendChild(card);
            });
        }
        
        standardResultsSection.classList.remove('hidden');
    }

    async function performRAGSearch(query) {
        const res = await fetch(`/api/ask?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        
        // Format markdown roughly (bolding)
        let formattedAnswer = data.answer.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formattedAnswer = formattedAnswer.replace(/\n/g, '<br>');
        
        ragAnswerContent.innerHTML = formattedAnswer;
        
        ragSourcesList.innerHTML = '';
        data.sources.forEach(source => {
            const li = document.createElement('li');
            li.textContent = source.title;
            ragSourcesList.appendChild(li);
        });
        
        ragAnswerSection.classList.remove('hidden');
    }
});
