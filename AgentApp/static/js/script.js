document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('userInput');
    const chatArea = document.getElementById('chatArea');

    // Allow Enter key to submit
    if (userInput) {
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }

    // Initialize Save Prediction button
    initSaveButton();
});

async function sendMessage() {
    const userInput = document.getElementById('userInput');
    const messageText = userInput.value.trim();

    if (!messageText) return;

    addMessage(messageText, 'user');
    userInput.value = '';

    const loadingId = addMessage('Analyzing market data...', 'bot');

    try {
        const response = await fetch('/api/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'),
            },
            body: JSON.stringify({
                prompt: messageText
            })
        });

        const data = await response.json();

        const loadingMsg = document.getElementById(loadingId);
        if (loadingMsg) loadingMsg.remove();

        if (response.ok) {
            addMessage(data.response, 'bot');
            if (data.data) {
                addTradeCard(data.data);
            }
            if (data.metrics) {
                updateMetricsUI(data.metrics);
            }
        } else {
            addMessage(`Error: ${data.error}`, 'bot');
        }

    } catch (error) {
        console.error('Error:', error);
        const loadingMsg = document.getElementById(loadingId);
        if (loadingMsg) loadingMsg.remove();
        addMessage('An error occurred while connecting to the server.', 'bot');
    }
}

function addMessage(text, sender) {
    const chatArea = document.getElementById('chatArea');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    messageDiv.id = 'msg-' + Date.now();

    const lines = String(text).split('\n');
    lines.forEach((line, idx) => {
        messageDiv.appendChild(document.createTextNode(line));
        if (idx < lines.length - 1) {
            messageDiv.appendChild(document.createElement('br'));
        }
    });

    chatArea.appendChild(messageDiv);
    chatArea.scrollTop = chatArea.scrollHeight;

    return messageDiv.id;
}

function addTradeCard(data) {
    const chatArea = document.getElementById('chatArea');
    const cardDiv = document.createElement('div');
    cardDiv.classList.add('trade-card');

    cardDiv.dataset.symbol = data.symbol;
    cardDiv.dataset.timeframe = data.timeframe;
    cardDiv.dataset.price = data.current_price;
    cardDiv.dataset.target = data.predicted_close;
    cardDiv.dataset.tp = data.tp;
    cardDiv.dataset.sl = data.sl;
    cardDiv.dataset.signal = data.signal;

    const signalClass = data.signal === 'BUY' ? 'signal-buy' : 'signal-sell';

    cardDiv.innerHTML = `
        <div class="trade-header">
            <span class="trade-pair" id="trade-pair">${data.symbol}</span>
            <span class="trade-signal ${signalClass}" id="trade-signal">${data.signal}</span>
        </div>
        <div class="trade-details trade-data">
            <div class="detail-row">
                <span class="label">Timeframe</span>
                <span class="value" id="timeframe">${data.timeframe}</span>
            </div>
            <div class="detail-row">
                <span class="label">Price</span>
                <span class="value" id="price">${data.current_price.toFixed(5)}</span>
            </div>
            <div class="detail-row">
                <span class="label">Target</span>
                <span class="value" id="target">${data.predicted_close.toFixed(5)}</span>
            </div>
            <div class="detail-row">
                <span class="label">TP</span>
                <span class="value" id="tp" style="color: var(--accent-color)">${data.tp.toFixed(5)}</span>
            </div>
            <div class="detail-row">
                <span class="label">SL</span>
                <span class="value" id="sl" style="color: var(--loss-color)">${data.sl.toFixed(5)}</span>
            </div>
        </div>
    `;

    chatArea.appendChild(cardDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function initSaveButton() {
    const saveBtn = document.getElementById('saveBtn');
    if (saveBtn) {
        saveBtn.addEventListener('click', async () => {
            const cards = document.querySelectorAll('.trade-card');
            if (!cards.length) {
                alert('No trade signal to save. Please run an analysis first.');
                return;
            }

            const latestCard = cards[cards.length - 1];
            const tradeData = {
                symbol: latestCard.dataset.symbol,
                timeframe: latestCard.dataset.timeframe,
                price: latestCard.dataset.price,
                target: latestCard.dataset.target,
                tp: latestCard.dataset.tp,
                sl: latestCard.dataset.sl,
                signal: latestCard.dataset.signal
            };

            try {
                const response = await fetch('/save_signal/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCookie('csrftoken')
                    },
                    body: JSON.stringify(tradeData)
                });

                const result = await response.json();
                if (result.status === 'success') {
                    saveBtn.textContent = 'Saved!';
                    saveBtn.disabled = true;
                    setTimeout(() => {
                        saveBtn.textContent = 'Save Prediction';
                        saveBtn.disabled = false;
                    }, 2000);
                } else {
                    alert('Error saving signal: ' + result.message);
                }
            } catch (error) {
                console.error('Error saving signal:', error);
            }
        });
    }
}


   

function updateMetricsUI(metrics) {
    const container = document.getElementById('metrics-container');
    const grid = document.getElementById('metricsGrid');
    
    if (!container || !grid) return;
    
    container.style.display = 'block';
    
    const metricItems = [
        { label: 'Hit Rate', value: (metrics.directional_accuracy * 100).toFixed(1) + '%', class: metrics.directional_accuracy > 0.5 ? 'positive' : '' },
        { label: 'F1 Score', value: metrics.f1_score.toFixed(2), class: metrics.f1_score > 0.5 ? 'positive' : '' },
        { label: 'Sharpe', value: metrics.sharpe_ratio.toFixed(2), class: metrics.sharpe_ratio > 1 ? 'positive' : '' },
        { label: 'Win Rate', value: (metrics.win_rate * 100).toFixed(1) + '%', class: metrics.win_rate > 0.5 ? 'positive' : '' },
        { label: 'RR Ratio', value: metrics.risk_reward.toFixed(2), class: metrics.risk_reward > 1.5 ? 'positive' : '' },
        { label: 'Expectancy', value: metrics.expectancy.toFixed(4), class: metrics.expectancy > 0 ? 'positive' : 'negative' }
    ];
    
    grid.innerHTML = metricItems.map(item => `
        <div class="metric-card">
            <span class="metric-label">${item.label}</span>
            <span class="metric-value ${item.class}">${item.value}</span>
        </div>
    `).join('');
    
    // Scroll to metrics
    container.scrollIntoView({ behavior: 'smooth' });
}

function getCookie(name) {
    const cookieValue = document.cookie
        .split('; ')
        .find(row => row.startsWith(name + '='))
        ?.split('=')[1];
    return cookieValue || '';
}
