#!/bin/bash
# Taeglicher REGIME-LIVE-Bot (Paper-Modus).
# Holt frische Tagesdaten fuer BTC + die Majors und laesst den Bot das Regime
# pruefen. Einmal taeglich per Cron, kurz nach 00:00 UTC (Tageskerze geschlossen).
#
# Installieren (einmalig, auf dem Server):
#   chmod +x /root/TradeBot/scripts/daily_regime.sh
#   ( crontab -l 2>/dev/null; echo "15 0 * * * /bin/bash /root/TradeBot/scripts/daily_regime.sh" ) | crontab -
#
# Mitlesen:   tail -f /root/trading-system/logs/regime_cron.log

set -u
cd /root/trading-system || exit 1
VENV=.venv/bin/python3
LOG=logs/regime_cron.log
MAJORS="BTC_USDT,BTC_USDC,ETH_USDC,SOL_USDC,BNB_USDC,XRP_USDC"

mkdir -p logs
echo "" >> "$LOG"
echo "===== $(date -u +'%Y-%m-%d %H:%M UTC') : taeglicher Regime-Check =====" >> "$LOG"
$VENV /root/TradeBot/scripts/collect_daily.py 400 "$MAJORS" >> "$LOG" 2>&1
$VENV /root/TradeBot/core/regime_live.py >> "$LOG" 2>&1
echo "----- fertig -----" >> "$LOG"
