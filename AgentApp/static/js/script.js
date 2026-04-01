document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('userInput');
    const chatArea = document.getElementById('chatArea');
    const pairDropdown = document.getElementById('pairDropdown');
    const timeframeDropdown = document.getElementById('timeframeDropdown');

    const startAnalysisBtn = document.getElementById('startAnalysisBtn');

    // Allow Enter key to submit
    if (userInput) {
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }

    if (startAnalysisBtn) {
        startAnalysisBtn.addEventListener('click', () => {
            const sym = pairDropdown.value;
            const tf = timeframeDropdown.value;
            // Clear previous metrics
            document.getElementById('metrics-container').style.display = 'none';
            sendMessage(false, `Perform a comprehensive technical analysis for ${sym} on the ${tf} timeframe. Include a clear signal (BUY/SELL/HOLD) and explain the reasoning based on the latest market data and your knowledge base.`);
        });
    }

    // Initialize Save Prediction button
    initSaveButton();
});

async function sendMessage(isAutoTrigged = false, customPrompt = null) {
    const userInput = document.getElementById('userInput');
    let messageText = customPrompt || userInput.value.trim();
    const pairDropdown = document.getElementById('pairDropdown');
    const timeframeDropdown = document.getElementById('timeframeDropdown');
    const accountBalanceInput = document.getElementById('accountBalance');
    
    const symbol = pairDropdown ? pairDropdown.value : 'EUR/USD';
    const timeframe = timeframeDropdown ? timeframeDropdown.value : '1h';
    const accountBalance = accountBalanceInput ? parseFloat(accountBalanceInput.value) : 10000.0;

    if (!messageText) return;

    addMessage(messageText, 'user');
    if (!customPrompt) userInput.value = '';

    const loadingId = addMessage('Analyzing market data...', 'bot');

    try {
        const response = await fetch('/api/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'),
            },
            body: JSON.stringify({
                prompt: messageText,
                symbol: symbol,
                timeframe: timeframe,
                account_balance: accountBalance
            })
        });

        const data = await response.json();

        const loadingMsg = document.getElementById(loadingId);
        if (loadingMsg) loadingMsg.remove();

        if (response.ok) {
            addMessage(data.response, 'bot', { symbol: data.data.symbol, timeframe: data.data.timeframe });
            if (data.data) {
                addTradeCard(data.data);
            }
            if (data.metrics) {
                updateMetricsUI(data.metrics);
            }
        } else {
            let errorMsg = data.error || 'An unknown error occurred.';
            addMessage(`❗ ${errorMsg}`, 'bot');
        }

    } catch (error) {
        console.error('Error:', error);
        const loadingMsg = document.getElementById(loadingId);
        if (loadingMsg) loadingMsg.remove();
        addMessage('An error occurred while connecting to the server.', 'bot');
    }
}

function addMessage(text, sender, meta = {}) {
    const chatArea = document.getElementById('chatArea');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    messageDiv.id = 'msg-' + Date.now();

    // Add metadata pills if present
    if (meta.symbol || meta.timeframe) {
        const metaDiv = document.createElement('div');
        metaDiv.classList.add('message-meta');
        
        if (meta.symbol) {
            const symPill = document.createElement('span');
            symPill.classList.add('meta-pill');
            symPill.textContent = meta.symbol;
            metaDiv.appendChild(symPill);
        }
        
        if (meta.timeframe) {
            const tfPill = document.createElement('span');
            tfPill.classList.add('meta-pill');
            tfPill.textContent = meta.timeframe;
            metaDiv.appendChild(tfPill);
        }
        
        messageDiv.appendChild(metaDiv);
    }

    const textSpan = document.createElement('span');
    const lines = String(text).split('\n');
    lines.forEach((line, idx) => {
        textSpan.appendChild(document.createTextNode(line));
        if (idx < lines.length - 1) {
            textSpan.appendChild(document.createElement('br'));
        }
    });
    messageDiv.appendChild(textSpan);

    chatArea.appendChild(messageDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
    
    if (sender === 'bot') {
        typingEffect(textSpan);
    }
    
    return messageDiv.id;
}

function typingEffect(element) {
    const text = element.innerHTML;
    element.innerHTML = '';
    let i = 0;
    const interval = setInterval(() => {
        if (i < text.length) {
            if (text[i] === '<') {
                const endTag = text.indexOf('>', i);
                element.innerHTML += text.substring(i, endTag + 1);
                i = endTag + 1;
            } else {
                element.innerHTML += text[i];
                i++;
            }
            // Auto-scroll chat area
            const chatArea = document.getElementById('chatArea');
            chatArea.scrollTop = chatArea.scrollHeight;
        } else {
            clearInterval(interval);
            // Highlight specific keywords for a 'premium' feel
            element.innerHTML = element.innerHTML
                .replace(/\b(BUY)\b/g, '<span class="signal-buy">$1</span>')
                .replace(/\b(SELL)\b/g, '<span class="signal-sell">$1</span>')
                .replace(/\b(HOLD)\b/g, '<span style="color: #888; font-weight: bold;">$1</span>');
        }
    }, 12);
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
        { label: 'F1 Score', value: (metrics.f1_score || 0).toFixed(2), class: (metrics.f1_score || 0) > 0.5 ? 'positive' : '' },
        { label: 'Sharpe', value: (metrics.sharpe_ratio || 0).toFixed(2), class: (metrics.sharpe_ratio || 0) > 1.5 ? 'positive' : '' },
        { label: 'Win Rate', value: (metrics.win_rate * 100).toFixed(1) + '%', class: metrics.win_rate > 0.5 ? 'positive' : '' },
        { label: 'RR Ratio', value: (metrics.risk_reward || 0).toFixed(2), class: (metrics.risk_reward || 0) > 2.0 ? 'positive' : '' },
        { label: 'Expectancy', value: (metrics.expectancy || 0).toFixed(4), class: (metrics.expectancy || 0) > 0 ? 'positive' : 'negative' }
    ];
    
    grid.innerHTML = metricItems.map(item => `
        <div class="metric-card">
            <span class="metric-label">${item.label}</span>
            <span class="metric-value ${item.class}">${item.value}</span>
        </div>
    `).join('');
    
    // Smooth scroll to metrics section
    setTimeout(() => {
        container.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }, 500);
}

function getCookie(name) {
    const cookieValue = document.cookie
        .split('; ')
        .find(row => row.startsWith(name + '='))
        ?.split('=')[1];
    return cookieValue || '';
}
