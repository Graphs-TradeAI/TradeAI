document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('userInput');
    const chatArea = document.getElementById('chatArea');

    // Allow Enter key to submit
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
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
            },
            body: JSON.stringify({
                prompt: messageText,
                api_key: ''
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

    // Convert newlines to breaks
    
    messageDiv.innerHTML = text.replace(/\n/g, '<br>');
    function typingEffect(messageDiv) {
        const text = messageDiv.textContent;
        let i = 0;
        const interval = setInterval(() => {
            messageDiv.textContent = text.substring(0, i) + '_';
            i++;
            if (i > text.length) {
                clearInterval(interval);
                messageDiv.textContent = text;
            }
        }, 100);
    }

    chatArea.appendChild(messageDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
    typingEffect(messageDiv);
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

function onSignIn(googleUser) {
    var profile = googleUser.getBasicProfile();
    console.log('Name: ' + profile.getName());
    console.log('Email: ' + profile.getEmail()); // This is null if the 'email' scope is not present.
}


document.getElementById("saveBtn").addEventListener("click", () => {
    const cards = document.querySelectorAll(".trade-card");

    if (!cards.length) {
        alert("No trade to save");
        return;
    }

    const latestCard = cards[cards.length - 1];

    const TradeData = {
        symbol: latestCard.dataset.symbol,
        timeframe: latestCard.dataset.timeframe,
        price: latestCard.dataset.price,
        target: latestCard.dataset.target,
        tp: latestCard.dataset.tp,
        sl: latestCard.dataset.sl,
        signal: latestCard.dataset.signal
    };

    console.log("Saving:", TradeData);

    fetch("/save_signal/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCookie("csrftoken")
        },
        body: JSON.stringify(TradeData)
    })
    .then(res => res.json())
    .then(data => console.log("Saved:", data))
    .catch(err => console.error(err));
});


   

