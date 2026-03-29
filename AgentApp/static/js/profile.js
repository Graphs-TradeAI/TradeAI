document.addEventListener('DOMContentLoaded', () => {
    // Initial fetch for settings only
    fetchProfileData(true);

    // Fetch Trades button
    const fetchTradesBtn = document.getElementById('fetchTradesBtn');
    if (fetchTradesBtn) {
        fetchTradesBtn.addEventListener('click', () => {
            const originalHTML = fetchTradesBtn.innerHTML;
            fetchTradesBtn.innerHTML = '<span>&#x21BB;</span> FETCHING...';
            fetchTradesBtn.disabled = true;
            
            fetchProfileData(false);
            
            setTimeout(() => {
                fetchTradesBtn.innerHTML = originalHTML;
                fetchTradesBtn.disabled = false;
            }, 1000);
        });
    }
    const sizeStrategy = document.getElementById('sizeStrategy');
    const fixedLotGroup = document.getElementById('fixedLotGroup');
    const riskPerTradeGroup = document.getElementById('riskPerTradeGroup');
    if (sizeStrategy) {
        sizeStrategy.addEventListener('change', () => {
            if (sizeStrategy.value === 'FIXED') {
                fixedLotGroup.style.display = 'block';
                riskPerTradeGroup.style.display = 'none';
            } else {
                fixedLotGroup.style.display = 'none';
                riskPerTradeGroup.style.display = 'block';
            }
        });
    }

    // Save Settings
    const saveSettingsBtn = document.getElementById('saveSettingsBtn');
    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener('click', updateProfileSettings);
    }
});

async function fetchProfileData(settingsOnly = false) {
    try {
        const response = await fetch('/get_profile_data/');
        const data = await response.json();
        
        if (response.ok) {
            updateProfileUI(data, settingsOnly);
        }
    } catch (error) {
        console.error('Error fetching profile:', error);
    }
}

function updateProfileUI(data, settingsOnly) {
    // Settings
    const s = data.settings;
    if (document.getElementById('autoTrading')) {
        document.getElementById('autoTrading').checked = s.auto_trading;
        document.getElementById('tradingMode').value = s.mode;
        document.getElementById('allowedSymbols').value = s.allowed_symbols;
        document.getElementById('allowedTimeframes').value = s.allowed_timeframes;
        document.getElementById('sizeStrategy').value = s.trade_size_strategy;
        document.getElementById('fixedLotSize').value = s.fixed_lot_size;
        document.getElementById('riskPerTrade').value = s.risk_per_trade;

        // Trigger size strategy change
        document.getElementById('sizeStrategy').dispatchEvent(new Event('change'));
    }

    if (settingsOnly) return;

    // Trades
    const tradesBody = document.getElementById('pastTradesBody');
    if (tradesBody) {
        if (data.trades.length === 0) {
            tradesBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #888;">No saved trades found.</td></tr>';
        } else {
            tradesBody.innerHTML = data.trades.map(t => `
                <tr>
                    <td>${t.timestamp}</td>
                    <td>${t.symbol}</td>
                    <td class="${t.signal === 'BUY' ? 'signal-buy' : 'signal-sell'}">${t.signal}</td>
                    <td>${t.price.toFixed(5)}</td>
                    <td style="color: var(--accent-color)">${t.tp.toFixed(5)}</td>
                    <td style="color: var(--loss-color)">${t.sl.toFixed(5)}</td>
                </tr>
            `).join('');
        }
    }
}

async function updateProfileSettings() {
    const settings = {
        auto_trading: document.getElementById('autoTrading').checked,
        mode: document.getElementById('tradingMode').value,
        allowed_symbols: document.getElementById('allowedSymbols').value,
        allowed_timeframes: document.getElementById('allowedTimeframes').value,
        trade_size_strategy: document.getElementById('sizeStrategy').value,
        fixed_lot_size: document.getElementById('fixedLotSize').value,
        risk_per_trade: document.getElementById('riskPerTrade').value
    };

    try {
        const response = await fetch('/update_profile_settings/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify(settings)
        });

        if (response.ok) {
            alert('Settings applied successfully!');
        } else {
            alert('Error updating settings.');
        }
    } catch (error) {
        console.error('Error updating settings:', error);
    }
}

function getCookie(name) {
    const cookieValue = document.cookie
        .split('; ')
        .find(row => row.startsWith(name + '='))
        ?.split('=')[1];
    return cookieValue || '';
}
