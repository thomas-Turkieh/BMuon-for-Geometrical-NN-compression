#!/usr/bin/env bash
# Fenêtre d'entraînement autorisée par la DSI :
#   - chaque soir de 18h30 à 7h30,
#   - le week-end : du vendredi 18h00 (accord DSI du 16 oct. 2026) au lundi 7h30 (sans interruption).
# Nos processus s'arrêtent d'eux-mêmes à 6h57 (consigne : tout doit être arrêté avant 6h59).
#
# window_secs_until_stop [DATE]
#   Affiche le nombre de secondes jusqu'au prochain arrêt de 6h57, ou rien (code 1) si DATE
#   (par défaut : maintenant) est hors fenêtre ou si il reste moins de 30 min.
#   DATE n'est utilisé que pour tester la fonction : window_secs_until_stop "2026-10-16 18:30".
window_secs_until_stop() {
  local ref="${1:-now}" now dow hhmm off day stop left
  now=$(date -d "$ref" +%s) || return 2
  dow=$(date -d "$ref" +%u)                    # 1 = lundi ... 7 = dimanche
  hhmm=$((10#$(date -d "$ref" +%H%M)))
  if [ "$dow" -ge 6 ]; then                    # samedi / dimanche : jusqu'au lundi matin
    off=$((8 - dow))
  elif [ "$hhmm" -lt 657 ]; then               # lundi..vendredi avant 6h57 : arrêt aujourd'hui
    off=0
  elif [ "$hhmm" -ge 1830 ] || { [ "$dow" -eq 5 ] && [ "$hhmm" -ge 1800 ]; }; then   # soir : vendredi (dès 18h00, accord DSI) -> lundi, sinon lendemain (dès 18h30)
    if [ "$dow" -eq 5 ]; then off=3; else off=1; fi
  else
    return 1                                   # jour de semaine entre 6h57 et 18h30
  fi
  day=$(date -d "$(date -d "$ref" +%F) +${off} days" +%F)       # date cible (sûr vis-à-vis du changement d'heure)
  stop=$(date -d "$day 06:57" +%s)
  left=$((stop - now))
  [ "$left" -ge 1800 ] || return 1
  echo "$left"
}
