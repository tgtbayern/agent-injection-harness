/**
 * GTO Poker Solver v3 — Interactive Poker Table
 *
 * Flow:
 *  1. Click a seat → mark as Hero, pick 2 hole cards
 *  2. Other seats are villains (click to toggle empty)
 *  3. Preflop action starts at UTG, goes clockwise
 *  4. For each player's turn, action buttons appear
 *  5. When it's Hero's turn → click "Get Advice"
 *  6. Postflop: pick board cards, continue actions
 */

const API = '';
const RANKS = 'AKQJT98765432';
const SUITS = ['s','h','d','c'];
const SYM = {s:'♠',h:'♥',d:'♦',c:'♣'};
const SCLS = {s:'s',h:'h',d:'d',c:'c'};
const POS = ['UTG','HJ','CO','BTN','SB','BB'];
const PREFLOP_ORDER = [0,1,2,3,4,5]; // UTG first preflop
const POSTFLOP_ORDER = [4,5,0,1,2,3]; // SB first postflop

const ACT_BG = {fold:'#dc2626',check:'#6b7280',call:'#16a34a',raise:'#7c3aed',bet:'#7c3aed',allin:'#ea580c'};

// ============================================================
// Game State
// ============================================================
let G = null; // will be set by resetAll()
let lastAdvice = null; // store last advice for villain popup

function makeState() {
    return {
        heroSeat: -1,           // which seat is hero (0-5, -1=none)
        seats: POS.map((p,i) => ({
            pos: p,
            active: true,       // in the hand
            folded: false,
            cards: [null,null],  // hero gets real cards, villains stay null
            stack: 100,
            betThisStreet: 0,
            isAllIn: false,
        })),
        street: 'preflop',      // preflop/flop/turn/river
        board: [],              // Card strings e.g. ['As','Kd','7h']
        pot: 1.5,               // BB posted
        currentBet: 1,          // biggest bet this street (BB=1 preflop)
        actionIndex: 0,         // index into action order
        actionOrder: [...PREFLOP_ORDER],
        history: [],            // [{seat,pos,action,amount}]
        phase: 'pick-hero',     // pick-hero | pick-cards | playing | done
        pickedCards: new Set(),
        pickerCallback: null,
        lastRaiser: -1,
        actionsThisRound: 0,
    };
}

function resetAll() {
    G = makeState();
    // Post blinds
    G.seats[4].betThisStreet = 0.5; // SB
    G.seats[4].stack = 99.5;
    G.seats[5].betThisStreet = 1;   // BB
    G.seats[5].stack = 99;
    G.pot = 1.5;
    G.currentBet = 1;
    renderAll();
}

// ============================================================
// Rendering
// ============================================================
function renderAll() {
    renderSeats();
    renderBoard();
    renderPot();
    renderActionPanel();
    renderLog();
}

function renderSeats() {
    for (let i = 0; i < 6; i++) {
        const el = document.getElementById('seat-'+i);
        const s = G.seats[i];

        el.classList.toggle('is-hero', i === G.heroSeat);
        el.classList.toggle('is-folded', s.folded || !s.active);
        el.classList.toggle('is-active', isCurrentTurn(i) && G.phase === 'playing');
        // Any non-hero active seat is clickable to view their range
        el.classList.toggle('clickable-villain',
            i !== G.heroSeat && G.heroSeat >= 0 && s.active && !s.folded && G.phase !== 'pick-hero');

        // Cards
        const cardsEl = document.getElementById('cards-'+i);
        if (i === G.heroSeat && (s.cards[0] || G.phase === 'pick-cards')) {
            let html = '';
            for (let j = 0; j < 2; j++) {
                const c = s.cards[j];
                if (c) {
                    html += miniCard(c);
                } else {
                    html += `<div class="mini-card empty-slot" onclick="openCardPicker(${i},${j})">+</div>`;
                }
            }
            cardsEl.innerHTML = html;
        } else if (s.active && !s.folded && G.phase !== 'pick-hero') {
            cardsEl.innerHTML = '<div class="mini-card facedown"></div><div class="mini-card facedown"></div>';
        } else {
            cardsEl.innerHTML = '';
        }

        // Stack
        document.getElementById('stack-'+i).textContent = s.stack.toFixed(1) + ' BB';

        // Status badge
        const statusEl = document.getElementById('status-'+i);
        const lastAct = getLastAction(i);
        if (lastAct) {
            const a = lastAct.action;
            statusEl.textContent = a === 'allin' ? 'ALL-IN' : a.toUpperCase();
            statusEl.className = 'seat-status ' + (a === 'raise' || a === 'bet' ? 'raised' : a);
        } else {
            statusEl.textContent = '';
            statusEl.className = 'seat-status';
        }

        // Bet display
        const betEl = document.getElementById('bet-'+i);
        if (s.betThisStreet > 0 && !s.folded) {
            betEl.textContent = s.betThisStreet.toFixed(1) + ' BB';
        } else {
            betEl.textContent = '';
        }
    }
}

function miniCard(c) {
    const r = c[0], s = c[1];
    return `<div class="mini-card ${SCLS[s]}"><span class="mc-rank">${r}</span><span class="mc-suit">${SYM[s]}</span></div>`;
}

function renderBoard() {
    const el = document.getElementById('boardCards');
    const label = document.getElementById('streetLabel');
    label.textContent = G.street.charAt(0).toUpperCase() + G.street.slice(1);

    if (G.street === 'preflop') {
        el.innerHTML = '';
        return;
    }

    const n = G.street === 'flop' ? 3 : G.street === 'turn' ? 4 : 5;
    let html = '';
    for (let i = 0; i < n; i++) {
        if (G.board[i]) {
            html += miniCard(G.board[i]);
        } else {
            html += `<div class="mini-card empty-slot" onclick="openBoardPicker(${i})">+</div>`;
        }
    }
    el.innerHTML = html;
}

function renderPot() {
    const total = G.pot + G.seats.reduce((s,p) => s + p.betThisStreet, 0);
    document.getElementById('potDisplay').textContent = 'Pot: ' + total.toFixed(1) + ' BB';
}

function renderActionPanel() {
    const prompt = document.getElementById('actionPrompt');
    const btns = document.getElementById('actionButtons');
    const slider = document.getElementById('raiseSliderContainer');

    slider.style.display = 'none';

    if (G.phase === 'pick-hero') {
        prompt.innerHTML = 'Click a seat to set as <strong>Hero</strong>. Right-click a seat to toggle empty.';
        prompt.style.display = '';
        btns.style.display = 'none';
        return;
    }

    if (G.phase === 'pick-cards') {
        prompt.innerHTML = 'Click the <strong>+</strong> cards to pick your hole cards.';
        prompt.style.display = '';
        btns.style.display = 'none';
        return;
    }

    if (G.phase === 'pick-board') {
        prompt.innerHTML = 'Click the <strong>+</strong> slots to deal board cards.';
        prompt.style.display = '';
        btns.style.display = 'none';
        return;
    }

    if (G.phase === 'done') {
        prompt.innerHTML = 'Hand complete. Click <strong>New Hand</strong> to start over.';
        prompt.style.display = '';
        btns.style.display = 'none';
        return;
    }

    // Playing phase
    const curSeat = getCurrentSeat();
    if (curSeat < 0) { G.phase = 'done'; renderAll(); return; }
    const s = G.seats[curSeat];
    const isHero = curSeat === G.heroSeat;
    const pos = s.pos;
    const toCall = Math.max(0, G.currentBet - s.betThisStreet);

    if (isHero) {
        prompt.innerHTML = `It's <strong>your turn</strong> (${pos}). Click <strong>Get Advice</strong> for GTO strategy, or choose an action.`;
    } else {
        prompt.innerHTML = `<strong>${pos}</strong>'s turn. Choose their action:`;
    }
    prompt.style.display = '';
    btns.style.display = 'flex';

    // Show/hide buttons based on context
    const foldBtn = btns.querySelector('.fold');
    const checkBtn = btns.querySelector('.check');
    const callBtn = btns.querySelector('.call');
    const betBtn = btns.querySelector('.bet');
    const callAmountEl = document.getElementById('callAmount');

    if (toCall > 0) {
        checkBtn.style.display = 'none';
        foldBtn.style.display = '';
        callBtn.style.display = '';
        callAmountEl.textContent = toCall.toFixed(1);
        betBtn.textContent = 'Raise';
    } else {
        checkBtn.style.display = '';
        foldBtn.style.display = 'none';
        callBtn.style.display = 'none';
        betBtn.textContent = 'Bet';
    }
}

function renderLog() {
    const el = document.getElementById('actionLog');
    if (G.history.length === 0) {
        el.innerHTML = '<div class="log-line muted">No actions yet</div>';
        return;
    }
    let html = '';
    let prevStreet = '';
    for (const h of G.history) {
        if (h.street && h.street !== prevStreet) {
            html += `<div class="log-line muted">── ${h.street} ──</div>`;
            prevStreet = h.street;
        }
        const aClass = h.action === 'raise' || h.action === 'bet' ? 'raise' : h.action;
        const amtStr = h.amount > 0 ? `<span class="log-amount">${h.amount.toFixed(1)} BB</span>` : '';
        html += `<div class="log-line">
            <span class="log-pos">${h.pos}</span>
            <span class="log-action ${aClass}">${h.action.toUpperCase()}</span>
            ${amtStr}
        </div>`;
    }
    el.innerHTML = html;
    el.scrollTop = el.scrollHeight;
}

// ============================================================
// Seat Click Handlers — moved to villain popup section below
// ============================================================

// ============================================================
// Card Picker
// ============================================================
function openCardPicker(seat, cardIdx) {
    G.pickerCallback = (cardStr) => {
        G.seats[seat].cards[cardIdx] = cardStr;
        G.pickedCards.add(cardStr);
        // If both cards picked, start playing
        if (G.seats[seat].cards[0] && G.seats[seat].cards[1]) {
            startPlaying();
        }
        renderAll();
    };
    document.getElementById('pickerTitle').textContent = `Pick card ${cardIdx + 1} for ${G.seats[seat].pos}`;
    showPickerGrid();
    document.getElementById('cardPickerOverlay').style.display = 'flex';
}

function openBoardPicker(slotIdx) {
    G.pickerCallback = (cardStr) => {
        G.board[slotIdx] = cardStr;
        G.pickedCards.add(cardStr);
        // Check if all needed board cards are picked
        const need = G.street === 'flop' ? 3 : G.street === 'turn' ? 4 : 5;
        const have = G.board.filter(c => c).length;
        if (have >= need) {
            G.phase = 'playing';
            G.actionOrder = [...POSTFLOP_ORDER];
            G.actionIndex = 0;
            G.actionsThisRound = 0;
            G.lastRaiser = -1;
            skipInactivePlayers();
        }
        renderAll();
    };
    document.getElementById('pickerTitle').textContent = `Pick board card ${slotIdx + 1}`;
    showPickerGrid();
    document.getElementById('cardPickerOverlay').style.display = 'flex';
}

function showPickerGrid() {
    const grid = document.getElementById('pickerGrid');
    let html = '';
    for (const suit of SUITS) {
        const isRed = suit === 'h' || suit === 'd';
        html += `<div class="picker-suit-row"><div class="picker-suit-label ${isRed?'red':'black'}">${SYM[suit]}</div><div class="picker-grid">`;
        for (const rank of RANKS) {
            const c = rank + suit;
            const picked = G.pickedCards.has(c);
            html += `<button type="button" class="pick-btn ${SCLS[suit]} ${picked?'picked':''}" onclick="pickCard('${c}')">
                <span>${rank}</span><span>${SYM[suit]}</span></button>`;
        }
        html += '</div></div>';
    }
    grid.innerHTML = html;
}

function pickCard(cardStr) {
    if (G.pickerCallback) {
        G.pickerCallback(cardStr);
        G.pickerCallback = null;
    }
    closePicker();
}

function closePicker() {
    document.getElementById('cardPickerOverlay').style.display = 'none';
}

// ============================================================
// Game Flow
// ============================================================
function startPlaying() {
    G.phase = 'playing';
    G.actionOrder = [...PREFLOP_ORDER];
    G.actionIndex = 0;
    G.actionsThisRound = 0;
    G.lastRaiser = -1;
    skipInactivePlayers();
}

function getCurrentSeat() {
    if (G.actionIndex >= G.actionOrder.length) return -1;
    return G.actionOrder[G.actionIndex];
}

function isCurrentTurn(seatIdx) {
    return G.phase === 'playing' && getCurrentSeat() === seatIdx;
}

function getLastAction(seatIdx) {
    // Get last action on current street
    for (let i = G.history.length - 1; i >= 0; i--) {
        if (G.history[i].seat === seatIdx && G.history[i].street === G.street) return G.history[i];
    }
    return null;
}

function skipInactivePlayers() {
    let safety = 0;
    while (safety < 12) {
        const cur = getCurrentSeat();
        if (cur < 0) break;
        const s = G.seats[cur];
        if (s.active && !s.folded && !s.isAllIn) break;
        G.actionIndex++;
        if (G.actionIndex >= G.actionOrder.length) break;
        safety++;
    }
}

function activePlayers() {
    return G.seats.filter(s => s.active && !s.folded).length;
}

function doAction(action) {
    const seatIdx = getCurrentSeat();
    if (seatIdx < 0) return;
    const s = G.seats[seatIdx];
    const toCall = Math.max(0, G.currentBet - s.betThisStreet);

    let amount = 0;

    if (action === 'fold') {
        s.folded = true;
    } else if (action === 'check') {
        // Nothing changes
    } else if (action === 'call') {
        amount = Math.min(toCall, s.stack);
        s.stack -= amount;
        s.betThisStreet += amount;
        if (s.stack <= 0) s.isAllIn = true;
    } else if (action === 'allin') {
        amount = s.stack;
        s.betThisStreet += amount;
        s.stack = 0;
        s.isAllIn = true;
        if (s.betThisStreet > G.currentBet) {
            G.currentBet = s.betThisStreet;
            G.lastRaiser = seatIdx;
            action = 'raise';
        } else {
            action = 'call';
        }
    }

    G.history.push({ seat: seatIdx, pos: s.pos, action, amount, street: G.street });
    G.actionsThisRound++;
    G.actionIndex++;

    // Check if round is over
    if (checkRoundOver()) {
        advanceStreet();
    } else {
        skipInactivePlayers();
        // Wrap around
        if (G.actionIndex >= G.actionOrder.length) {
            G.actionIndex = 0;
            skipInactivePlayers();
            if (checkRoundOver()) advanceStreet();
        }
    }

    // Check if hand is over (1 player left)
    if (activePlayers() <= 1) {
        G.phase = 'done';
    }

    renderAll();
}

function doRaise(totalAmount) {
    const seatIdx = getCurrentSeat();
    if (seatIdx < 0) return;
    const s = G.seats[seatIdx];

    const putIn = totalAmount - s.betThisStreet;
    const actual = Math.min(putIn, s.stack);
    s.stack -= actual;
    s.betThisStreet += actual;
    if (s.stack <= 0) s.isAllIn = true;

    G.currentBet = s.betThisStreet;
    G.lastRaiser = seatIdx;

    const action = G.currentBet > 1 || G.street !== 'preflop' ? 'raise' : 'bet';
    G.history.push({ seat: seatIdx, pos: s.pos, action, amount: s.betThisStreet, street: G.street });
    G.actionsThisRound++;

    // Reset action - everyone after raiser needs to act again
    const order = G.street === 'preflop' ? [...PREFLOP_ORDER] : [...POSTFLOP_ORDER];
    G.actionOrder = order;
    // Start from the player after the raiser
    const rIdx = order.indexOf(seatIdx);
    G.actionIndex = rIdx + 1;
    if (G.actionIndex >= order.length) G.actionIndex = 0;

    skipInactivePlayers();
    if (checkRoundOver()) advanceStreet();
    if (activePlayers() <= 1) G.phase = 'done';

    renderAll();
}

function checkRoundOver() {
    const active = G.seats.filter(s => s.active && !s.folded && !s.isAllIn);
    if (active.length === 0) return true;
    if (active.length === 1 && G.seats.filter(s => s.active && !s.folded).length === 1) return true; // everyone else folded

    // All active (non-allin) players have matched current bet and had a chance to act
    if (G.actionsThisRound < activePlayers()) return false;

    const allMatched = active.every(s => Math.abs(s.betThisStreet - G.currentBet) < 0.01 || s.isAllIn);
    return allMatched;
}

function advanceStreet() {
    // Collect bets into pot
    for (const s of G.seats) {
        G.pot += s.betThisStreet;
        s.betThisStreet = 0;
    }
    G.currentBet = 0;
    G.lastRaiser = -1;
    G.actionsThisRound = 0;

    const streets = ['preflop','flop','turn','river'];
    const idx = streets.indexOf(G.street);
    if (idx >= 3) { G.phase = 'done'; return; }

    G.street = streets[idx + 1];
    G.actionOrder = [...POSTFLOP_ORDER];
    G.actionIndex = 0;

    // Need board cards
    const need = G.street === 'flop' ? 3 : G.street === 'turn' ? 4 : 5;
    const have = G.board.filter(c => c).length;
    if (have < need) {
        G.phase = 'pick-board';
    } else {
        skipInactivePlayers();
    }
}

// ============================================================
// Raise Slider
// ============================================================
function showRaiseSlider() {
    const seatIdx = getCurrentSeat();
    const s = G.seats[seatIdx];
    const toCall = Math.max(0, G.currentBet - s.betThisStreet);
    const minRaise = G.currentBet + Math.max(1, G.currentBet); // min 2x current bet
    const maxRaise = s.betThisStreet + s.stack;

    document.getElementById('actionButtons').style.display = 'none';
    const container = document.getElementById('raiseSliderContainer');
    container.style.display = '';

    const slider = document.getElementById('raiseSlider');
    const input = document.getElementById('raiseInput');
    slider.min = Math.ceil(minRaise);
    slider.max = Math.floor(maxRaise);
    slider.value = Math.min(Math.ceil(minRaise * 1.5), Math.floor(maxRaise));
    input.value = slider.value;
    input.min = slider.min;
    input.max = slider.max;

    slider.oninput = () => { input.value = slider.value; };
    input.oninput = () => { slider.value = input.value; };

    // Presets
    const pot = G.pot + G.seats.reduce((a,p) => a + p.betThisStreet, 0);
    const presetsEl = document.getElementById('raisePresets');
    let phtml = '';
    if (G.street === 'preflop') {
        [2.5, 3, 4, 5].forEach(x => {
            if (x <= maxRaise) phtml += `<button type="button" onclick="setRaise(${x})">${x} BB</button>`;
        });
        const open3x = G.currentBet * 3;
        if (G.currentBet > 1 && open3x <= maxRaise) {
            phtml += `<button type="button" onclick="setRaise(${open3x.toFixed(1)})">3x (${open3x.toFixed(1)})</button>`;
        }
    } else {
        [0.33, 0.5, 0.67, 0.75, 1.0, 1.5].forEach(f => {
            const amt = G.currentBet + pot * f;
            if (amt <= maxRaise && amt >= minRaise) {
                phtml += `<button type="button" onclick="setRaise(${amt.toFixed(1)})">${Math.round(f*100)}% pot</button>`;
            }
        });
    }
    phtml += `<button type="button" onclick="setRaise(${maxRaise.toFixed(1)})">All-in</button>`;
    presetsEl.innerHTML = phtml;
}

function setRaise(val) {
    document.getElementById('raiseSlider').value = val;
    document.getElementById('raiseInput').value = val;
}

function confirmRaise() {
    const val = parseFloat(document.getElementById('raiseInput').value);
    if (isNaN(val) || val < 1) { showToast('Invalid raise amount','error'); return; }
    doRaise(val);
    document.getElementById('raiseSliderContainer').style.display = 'none';
}

function cancelRaise() {
    document.getElementById('raiseSliderContainer').style.display = 'none';
    document.getElementById('actionButtons').style.display = 'flex';
}

// ============================================================
// GTO Advice
// ============================================================
async function getAdvice() {
    if (G.heroSeat < 0 || !G.seats[G.heroSeat].cards[0]) {
        showToast('Set hero seat and pick cards first','error');
        return;
    }

    const btn = document.getElementById('getAdviceBtn');
    btn.disabled = true;
    btn.textContent = 'Solving...';

    try {
        const hero = G.seats[G.heroSeat];
        const handStr = hero.cards[0] + hero.cards[1];
        const boardStr = G.board.filter(c => c).join(' ');
        const nVillains = G.seats.filter(s => s.active && !s.folded && s !== hero).length;

        // Build action history in format backend expects
        const actionHistory = G.history.map(h => `${h.pos}:${h.action}${h.amount > 0 ? '_'+h.amount.toFixed(1) : ''}`);

        const pot = G.pot + G.seats.reduce((a,p) => a + p.betThisStreet, 0);

        const body = {
            hero_hand: handStr,
            hero_position: hero.pos,
            board: boardStr,
            pot_size: pot,
            stack_size: hero.stack,
            action_history: actionHistory,
            street: G.street,
            num_villains: nVillains,
        };

        const res = await fetch(`${API}/api/advisor`, {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error((await res.json()).detail || 'Failed');
        const data = await res.json();
        lastAdvice = data;
        displayAdvice(data);
        markVillainSeatsClickable();
    } catch(e) {
        showToast('Error: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Get Advice';
    }
}

function displayAdvice(data) {
    const body = document.getElementById('adviceBody');
    const actions = data.recommended_actions || {};
    const sorted = Object.entries(actions).sort((a,b) => b[1] - a[1]);
    const hero = G.seats[G.heroSeat];
    const heroLabel = hero.cards.map(c => c ? c[0] + SYM[c[1]] : '?').join(' ');
    const nv = data.num_villains || 1;

    let html = '';

    // Badges
    html += '<div class="adv-scenario">';
    html += `<span class="adv-badge street">${data.scenario_type || G.street}</span>`;
    if (nv >= 2) html += `<span class="adv-badge multiway">${nv+1}-WAY</span>`;
    if (data.hand_tier) html += `<span class="adv-badge tier">Tier ${data.hand_tier}</span>`;
    if (data.hand) html += `<span class="adv-badge" style="background:#334">${data.hand}</span>`;
    html += '</div>';

    // Hero hand
    html += `<div class="adv-hand-display">${hero.cards.map(c => c ? miniCard(c) : '').join('')}</div>`;

    // Combined bar
    html += '<div class="adv-combined">';
    for (const [act, freq] of sorted) {
        if (freq < 0.005) continue;
        const pct = (freq*100).toFixed(1);
        const bg = getActionBg(act);
        const label = getActionLabel(act);
        html += `<div class="adv-seg" style="width:${pct}%;background:${bg}">
            ${freq > 0.12 ? label + ' ' + Math.round(freq*100) + '%' : ''}
        </div>`;
    }
    html += '</div>';

    // Individual bars
    for (const [act, freq] of sorted) {
        if (freq < 0.005) continue;
        const pct = (freq*100).toFixed(1);
        const bg = getActionBg(act);
        const label = getActionLabel(act);
        const star = act === sorted[0][0] ? '★ ' : '';
        html += `<div class="adv-bar">
            <div class="adv-bar-head"><span class="adv-bar-name" style="color:${bg}">${star}${label}</span><span class="adv-bar-pct" style="color:${bg}">${pct}%</span></div>
            <div class="adv-bar-track"><div class="adv-bar-fill" style="width:${pct}%;background:${bg}">${freq>.15?pct+'%':''}</div></div>
        </div>`;
    }

    // ==== 📐 GTO ANALYSIS (immediately after action bars) ====
    if (data.gto_reasoning) {
        const gr = data.gto_reasoning;

        // Dynamic analysis lines
        if (gr.analysis && gr.analysis.length) {
            html += '<div class="gto-math">';
            html += '<div class="gto-math-title">📐 Hand-Specific Analysis</div>';
            for (const line of gr.analysis) {
                const rendered = line.replace(/\*\*(.+?)\*\*/g, '<strong style="color:var(--text)">$1</strong>');
                html += `<div class="gto-math-line" style="border-bottom:1px solid rgba(255,255,255,.04);padding:4px 0">${rendered}</div>`;
            }
            html += '</div>';
        }

        // Math calculations
        if (gr.math && gr.math.length) {
            html += '<div class="gto-math" style="margin-top:6px">';
            html += '<div class="gto-math-title">🧮 GTO Math</div>';
            for (const line of gr.math) {
                const rendered = line.replace(/\*\*(.+?)\*\*/g, '<strong style="color:var(--text)">$1</strong>');
                html += `<div class="gto-math-line">${rendered}</div>`;
            }
            html += '</div>';
        }

        // ==== 👥 VILLAIN — click seats to see detail ====
        const vh = gr.villain_hands;
        const villainPositions = vh ? Object.keys(vh) : [];
        if (villainPositions.length > 0) {
            html += '<div class="vr-section" style="text-align:center;padding:16px">';
            html += '<div class="vr-title">👥 Click a villain seat (yellow dashed border) to see their response</div>';
            html += '<div style="display:flex;gap:6px;justify-content:center;flex-wrap:wrap;margin-top:8px">';
            for (const vp of villainPositions) {
                const vd = vh[vp] || {};
                const r = vd.raise || {pct:0};
                const c = vd.call || {pct:0};
                const f = vd.fold || {pct:0};
                html += `<button type="button" onclick="showVillainPopup('${vp}')" style="background:var(--bg3);border:1px dashed var(--yellow);color:var(--text);padding:8px 14px;border-radius:8px;cursor:pointer;font-size:12px;font-weight:700">`;
                html += `${vp}<br><span style="font-size:10px;color:var(--text3)">R:${r.pct?.toFixed(0)||0}% C:${c.pct?.toFixed(0)||0}% F:${f.pct?.toFixed(0)||0}%</span>`;
                html += `</button>`;
            }
            html += '</div></div>';
        }
    }

    // ==== Raise sizes ====
    if (data.raise_sizes && data.raise_sizes.length > 0) {
        html += '<div class="adv-section-title">📐 Raise Sizing</div>';
        for (const rs of data.raise_sizes) {
            html += `<div class="adv-raise-item"><div><div class="adv-raise-label">${rs.label||rs.size+'x'}</div><div class="adv-raise-desc">${rs.description||''}</div></div><div class="adv-raise-freq">${rs.frequency?(rs.frequency*100).toFixed(0)+'%':''}</div></div>`;
        }
    }

    // ==== Equity ====
    if (data.equity != null) {
        const eq = data.equity;
        const ep = (eq*100).toFixed(1);
        const ec = eq >= .6 ? '#22c55e' : eq >= .4 ? '#eab308' : '#ef4444';
        html += `<div class="adv-equity"><span style="font-size:11px;color:var(--text3)">EQ</span><div class="adv-eq-bar"><div class="adv-eq-fill" style="width:${ep}%;background:${ec}"></div></div><span class="adv-eq-val" style="color:${ec}">${ep}%</span></div>`;
        if (data.equity_detail && data.equity_detail.num_villains >= 2) {
            const ed = data.equity_detail;
            html += '<div class="adv-grid"><div class="adv-grid-item"><div class="adv-grid-label">Win</div><div class="adv-grid-val" style="color:#22c55e">'+(ed.win_pct*100).toFixed(1)+'%</div></div>';
            html += '<div class="adv-grid-item"><div class="adv-grid-label">Lose</div><div class="adv-grid-val" style="color:#ef4444">'+(ed.lose_pct*100).toFixed(1)+'%</div></div></div>';
        }
    }

    // ==== Collapsible: Hand analysis + Bet sizes ====
    if (data.hand_analysis || (data.available_bet_sizes && G.street !== 'preflop')) {
        html += `<details style="margin-top:10px"><summary style="cursor:pointer;color:var(--text2);font-size:12px;font-weight:600">🔍 Details (hand analysis, bet sizes)</summary>`;
        if (data.hand_analysis) {
            const ha = data.hand_analysis;
            html += '<div class="adv-grid" style="margin-top:8px">';
            html += `<div class="adv-grid-item"><div class="adv-grid-label">Made Hand</div><div class="adv-grid-val">${fmtMadeHand(ha.made_hand)}</div></div>`;
            html += `<div class="adv-grid-item"><div class="adv-grid-label">Category</div><div class="adv-grid-val" style="color:${catColor(ha.category)}">${(ha.category||'').toUpperCase()}</div></div>`;
            html += `<div class="adv-grid-item"><div class="adv-grid-label">Draw</div><div class="adv-grid-val">${ha.draw?fmtDraw(ha.draw,ha.draw_outs):'None'}</div></div>`;
            html += `<div class="adv-grid-item"><div class="adv-grid-label">Strength</div><div class="adv-grid-val">${(ha.relative_strength*100).toFixed(0)}%</div></div>`;
            html += '</div>';
        }
        if (data.available_bet_sizes && data.available_bet_sizes.length > 0 && G.street !== 'preflop') {
            html += '<div class="adv-section-title">📊 Bet Sizes</div><div class="adv-grid">';
            for (const bs of data.available_bet_sizes.slice(0,6)) {
                html += `<div class="adv-grid-item"><div class="adv-grid-label">${bs.label}</div><div class="adv-grid-val">${bs.amount} BB</div></div>`;
            }
            html += '</div>';
        }
        html += '</details>';
    }

    body.innerHTML = html;
}

// ============================================================
// Helpers
// ============================================================
function getActionBg(a) {
    if (a.startsWith('raise')||a.startsWith('bet')) return '#7c3aed';
    return ACT_BG[a]||'#6b7280';
}
function getActionLabel(a) {
    if (a==='fold') return 'Fold';
    if (a==='check') return 'Check';
    if (a==='call') return 'Call';
    if (a==='raise') return 'Raise';
    if (a==='allin') return 'All-in';
    if (a.startsWith('raise_')) return 'Raise '+a.split('_')[1]+'%';
    if (a.startsWith('bet_')) return 'Bet '+a.split('_')[1]+'%';
    return a;
}
function fmtMadeHand(m) {
    const map = {top_pair_top_kicker:'TPTK',top_pair_good_kicker:'TP Good K',top_pair_weak_kicker:'TP Weak K',overpair:'Overpair',two_pair:'Two Pair',set:'Set',trips:'Trips',straight:'Straight',flush:'Flush',full_house:'Full House',quads:'Quads',straight_flush:'Str Flush',second_pair:'2nd Pair',bottom_pair:'Bottom Pair',pocket_pair:'Pocket Pair',ace_high:'Ace High',high_card:'High Card'};
    return map[m]||(m||'').replace(/_/g,' ');
}
function fmtDraw(d,o) {
    const map = {flush_draw:'Flush Draw',straight_draw:'Straight Draw',combo_draw:'Combo Draw'};
    return `${map[d]||d} (${o} outs)`;
}
function catColor(c) {
    return {nuts:'#eab308',strong:'#22c55e',medium:'#3b82f6',weak:'#ef4444',air:'#dc2626',premium:'#eab308',unknown:'#6b7280'}[c]||'#6b7280';
}

function showToast(msg,type='error') {
    const c = document.getElementById('toastContainer');
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.innerHTML = `<span>${type==='error'?'❌':'✅'}</span><span>${msg}</span>`;
    c.appendChild(t);
    setTimeout(()=>{t.style.opacity='0';setTimeout(()=>t.remove(),300)},3500);
}

// ============================================================
// Villain Seat Click → Show their response popup
// ============================================================
function markVillainSeatsClickable() {
    document.querySelectorAll('.seat').forEach((el, i) => {
        el.classList.remove('clickable-villain');
    });
    if (!lastAdvice || !lastAdvice.gto_reasoning) return;
    const vh = lastAdvice.gto_reasoning.villain_hands || {};
    let count = 0;
    for (let i = 0; i < 6; i++) {
        if (i === G.heroSeat) continue;
        const pos = POS[i];
        if (vh[pos]) {
            document.getElementById('seat-'+i).classList.add('clickable-villain');
            count++;
        }
    }
    if (count > 0) {
        showToast(`Click on opponent seats (yellow border) to see their response matrix`, 'success');
    }
}

// Attach click handler to seats for villain popup
document.querySelectorAll('.seat').forEach((el, i) => {
    el.addEventListener('click', (e) => {
        e.preventDefault();
        if (G.phase === 'pick-hero') {
            G.heroSeat = i;
            G.phase = 'pick-cards';
            renderAll();
            return;
        }
        // Any time after hero is set: clicking a non-hero seat → show their range
        if (i !== G.heroSeat && G.heroSeat >= 0 && G.seats[i].active && !G.seats[i].folded) {
            const pos = POS[i];
            // If we already have advice data for this villain, use it
            if (lastAdvice && lastAdvice.gto_reasoning && lastAdvice.gto_reasoning.villain_hands && lastAdvice.gto_reasoning.villain_hands[pos]) {
                showVillainPopup(pos);
            } else {
                // Otherwise fetch it live
                fetchVillainRange(pos);
            }
        }
    });
    el.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        if (G.phase === 'pick-hero' || G.phase === 'pick-cards') {
            G.seats[i].active = !G.seats[i].active;
            if (i === G.heroSeat) { G.heroSeat = -1; G.phase = 'pick-hero'; }
            renderAll();
        }
    });
});

async function fetchVillainRange(pos) {
    if (G.heroSeat < 0 || !G.seats[G.heroSeat].cards[0]) {
        showToast('Pick your hero cards first', 'error');
        return;
    }
    const hero = G.seats[G.heroSeat];
    const handStr = hero.cards[0] + hero.cards[1];
    const nBoard = G.street === 'river' ? 5 : G.street === 'turn' ? 4 : G.street === 'flop' ? 3 : 0;
    const boardStr = G.board.slice(0, nBoard).filter(c => c).map(c => c[0] + c[1]).join(' ');
    const nVillains = G.seats.filter(s => s.active && !s.folded && s !== hero).length;
    const actionHistory = G.history.map(h => `${h.pos}:${h.action}${h.amount > 0 ? '_'+h.amount.toFixed(1) : ''}`);
    const pot = G.pot + G.seats.reduce((a, p) => a + p.betThisStreet, 0);

    showToast('Loading range...', 'success');
    try {
        const res = await fetch(`${API}/api/advisor`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                hero_hand: handStr,
                hero_position: hero.pos,
                board: boardStr,
                pot_size: pot,
                stack_size: hero.stack,
                action_history: actionHistory,
                street: G.street,
                num_villains: nVillains,
            }),
        });
        if (!res.ok) throw new Error('API error');
        const data = await res.json();
        lastAdvice = data;
        if (data.gto_reasoning && data.gto_reasoning.villain_hands && data.gto_reasoning.villain_hands[pos]) {
            showVillainPopup(pos);
        } else {
            showToast('No range data for ' + pos, 'error');
        }
    } catch(e) {
        showToast('Error: ' + e.message, 'error');
    }
}

function showVillainPopup(pos) {
    if (!lastAdvice || !lastAdvice.gto_reasoning) return;
    const vh = lastAdvice.gto_reasoning.villain_hands || {};
    const vData = vh[pos];
    if (!vData) return;

    document.getElementById('villainPopupTitle').textContent = pos + " — GTO Response Matrix";
    const body = document.getElementById('villainPopupBody');

    let html = '';

    // Range description
    if (vData.range_name) {
        html += `<div style="font-size:13px;color:var(--text2);margin-bottom:6px">Range: <strong style="color:var(--text)">${vData.range_name}</strong></div>`;
    }
    if (vData.scenario) {
        const s = vData.scenario.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html += `<div style="font-size:12px;color:var(--text3);margin-bottom:4px">${s}</div>`;
    }
    if (vData.mdf) {
        html += `<div style="font-size:11px;color:var(--text3);margin-bottom:8px">MDF: ${(vData.mdf*100).toFixed(0)}% | Pot odds: ${(vData.pot_odds*100).toFixed(1)}%</div>`;
    }

    // Summary bar
    const r = vData.raise||{pct:0}, c = vData.call||{pct:0}, f = vData.fold||{pct:0};
    html += '<div class="vp-range-bar">';
    if (r.pct > 0) html += `<div class="vp-range-seg" style="width:${r.pct}%;background:#7c3aed">${r.pct>10?'Raise '+r.pct.toFixed(0)+'%':''}</div>`;
    if (c.pct > 0) html += `<div class="vp-range-seg" style="width:${c.pct}%;background:#16a34a">${c.pct>10?'Call '+c.pct.toFixed(0)+'%':''}</div>`;
    if (f.pct > 0) html += `<div class="vp-range-seg" style="width:${f.pct}%;background:#dc2626">${f.pct>10?'Fold '+f.pct.toFixed(0)+'%':''}</div>`;
    html += '</div>';

    // Legend
    html += `<div style="display:flex;gap:12px;margin:8px 0;font-size:11px;flex-wrap:wrap">
        <span><span style="display:inline-block;width:12px;height:12px;background:#7c3aed;border-radius:2px;vertical-align:middle"></span> Raise/Bet</span>
        <span><span style="display:inline-block;width:12px;height:12px;background:#16a34a;border-radius:2px;vertical-align:middle"></span> Call/Check</span>
        <span><span style="display:inline-block;width:12px;height:12px;background:#dc2626;border-radius:2px;vertical-align:middle"></span> Fold/Check behind</span>
        <span><span style="display:inline-block;width:12px;height:12px;background:var(--bg3);border-radius:2px;vertical-align:middle"></span> Not in range</span>
    </div>`;

    // 13×13 MATRIX
    const matrix = vData.matrix;
    if (matrix) {
        html += '<div class="range-matrix" style="gap:2px;max-width:100%">';
        const R = 'AKQJT98765432';
        for (let i = 0; i < 13; i++) {
            for (let j = 0; j < 13; j++) {
                let hand;
                if (i === j) hand = R[i] + R[j];
                else if (i < j) hand = R[i] + R[j] + 's';
                else hand = R[j] + R[i] + 'o';

                const m = matrix[hand];
                let bg, color;
                if (!m || !m.in_range) {
                    bg = 'var(--bg3)';
                    color = 'var(--text3)';
                } else if (m.raise >= 0.5) {
                    // Mostly raise
                    bg = `rgba(124,58,237,${0.3 + m.raise * 0.7})`;
                    color = '#fff';
                } else if (m.call >= 0.5) {
                    // Mostly call
                    bg = `rgba(22,163,74,${0.3 + m.call * 0.7})`;
                    color = '#fff';
                } else if (m.fold >= 0.5) {
                    // Mostly fold
                    bg = `rgba(239,68,68,${0.15 + m.fold * 0.3})`;
                    color = 'rgba(255,255,255,0.6)';
                } else if (m.raise > 0 && m.call > 0) {
                    // Mixed raise/call
                    bg = `linear-gradient(135deg, rgba(124,58,237,0.7) ${m.raise*100}%, rgba(22,163,74,0.7) ${m.raise*100}%)`;
                    color = '#fff';
                } else {
                    bg = 'var(--bg3)';
                    color = 'var(--text3)';
                }

                const tooltip = m && m.in_range
                    ? `title="R:${(m.raise*100).toFixed(0)}% C:${(m.call*100).toFixed(0)}% F:${(m.fold*100).toFixed(0)}%"`
                    : 'title="Not in range"';

                html += `<div class="range-cell" style="background:${bg};color:${color};min-height:28px;font-size:9px" ${tooltip}>
                    <span class="cell-label">${hand}</span>
                </div>`;
            }
        }
        html += '</div>';
    }

    body.innerHTML = html;
    document.getElementById('villainPopup').style.display = 'flex';
}

function closeVillainPopup(e) {
    if (e && e.target !== document.getElementById('villainPopup')) return;
    document.getElementById('villainPopup').style.display = 'none';
}

// ============================================================
// Preflop Range Chart
// ============================================================
function toggleRangeChart() {
    const el = document.getElementById('rangeOverlay');
    if (el.style.display === 'none') {
        el.style.display = 'flex';
        buildRangeMatrix();
        loadRange('UTG', document.querySelector('#rangePosSel .position-btn'));
    } else {
        el.style.display = 'none';
    }
}

function buildRangeMatrix() {
    const m = document.getElementById('rangeMatrix');
    m.innerHTML = '';
    for (let i = 0; i < 13; i++) {
        for (let j = 0; j < 13; j++) {
            const r1 = RANKS[i], r2 = RANKS[j];
            let hand;
            if (i===j) hand = r1+r2;
            else if (i<j) hand = r1+r2+'s';
            else hand = r2+r1+'o';
            const c = document.createElement('div');
            c.className = 'range-cell';
            c.dataset.hand = hand;
            c.innerHTML = `<span class="cell-label">${hand}</span>`;
            c.style.background = 'var(--bg3)';
            c.style.color = 'var(--text3)';
            m.appendChild(c);
        }
    }
}

async function loadRange(pos, btnEl) {
    document.querySelectorAll('#rangePosSel .position-btn').forEach(b=>b.classList.remove('active'));
    if(btnEl) btnEl.classList.add('active');
    try {
        const res = await fetch(`${API}/api/preflop/${pos}`);
        if (!res.ok) return;
        const data = await res.json();
        document.getElementById('rangeStats').innerHTML =
            `<span>Open: <b>${data.open_percentage}%</b></span>` +
            `<span>Position: <b>${pos}</b></span>`;
        document.querySelectorAll('.range-cell').forEach(cell => {
            const hand = cell.dataset.hand;
            const s = data.strategies[hand] || {fold:1};
            const fold = s.fold||0, raise = s.raise||0, call = s.call||0;
            const open = 1 - fold;
            if (open <= 0.01) {
                cell.style.background = 'var(--bg3)'; cell.style.color = 'var(--text3)';
            } else if (raise > 0 && call > 0) {
                cell.style.background = `linear-gradient(135deg,#7c3aed ${Math.round(raise*100)}%,#16a34a ${Math.round(raise*100)}%)`;
                cell.style.color = '#fff';
            } else if (raise > 0) {
                cell.style.background = open>=.99?'#3b82f6':`rgba(59,130,246,${open*.9+.1})`;
                cell.style.color = open>.4?'#fff':'#93c5fd';
            } else if (call > 0) {
                cell.style.background = open>=.99?'#16a34a':`rgba(22,163,74,${open*.9+.1})`;
                cell.style.color = open>.4?'#fff':'#86efac';
            }
        });
    } catch(e) { /* ignore */ }
}

// ============================================================
// Init
// ============================================================
resetAll();
