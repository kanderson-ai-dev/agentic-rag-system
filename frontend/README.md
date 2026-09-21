# Frontend

Puro HTML/CSS/JS — sin framework, sin build step, sin dependencias externas.
Servido por FastAPI vía `StaticFiles` montado en `/` (ver `app/main.py`).

## Estructura

| Archivo      | Responsabilidad |
| ------------ | ------------------------------------------------------------------ |
| `index.html` | Marcado semántico + script inline que aplica el tema antes del primer *paint* (anti-FOUC). |
| `styles.css` | Design system: tokens, reset, componentes base, temas claro/oscuro, responsive. |
| `app.js`     | Lógica: auth gate, chat, modal de revisión humana (HITL), dashboard, toggle de tema. |

## Design system (tokens)

Todo valor visual deriva de **custom properties** (variables CSS); no hay valores
"mágicos" hardcodeados en los componentes.

- **Paleta** (`--bg`, `--surface*`, `--border*`, `--text*`, `--accent*`, `--success`,
  `--danger`, `--warning`).
- **Tipografía** (`--font-sans`, `--font-mono`, escala `--text-xs`…`--text-3xl`,
  `--leading-*`, `--weight-*`).
- **Espaciado** base 4px (`--space-1`…`--space-7`).
- **Radios** (`--radius-sm/md/lg/full`), **sombras** (`--shadow-sm/md/lg`) y
  **transiciones** (`--transition-fast`, `--transition`).

### Temas

Dos temas definidos como bloques de tokens: `[data-theme="dark"]` (por defecto) y
`[data-theme="light"]`. El toggle en el header:

1. Aplica el tema guardado en `localStorage` (`key = "theme"`) antes del primer
   *paint* mediante un script inline en `<head>`.
2. Si no hay preferencia guardada, respeta `prefers-color-scheme`.
3. Al clickear, alterna el `data-theme` del `<html>` y persiste en `localStorage`.

## Componentes base reutilizables

- **Botones**: `.btn` (primario) y variantes `.secondary`, `.danger`, `.ghost`, con
  estados `:hover`, `:active`, `:disabled` y foco visible.
- **Inputs / textarea**: estados `:focus` con anillo de acento.
- **Cards**: `.card`.
- **Badges**: `.badge` (+ `.signed-in`).
- **Tablas**: `table.recent`.
- **Stats**: `.stat` (+ `.quality-stat.pass/.fail`).

## Convenciones

- El contenido dinámico se inserta con `textContent` (o `innerHTML` solo para
  estructura estática conocida); **nunca** se renderiza salida del LLM sin sanear.
- El JWT se guarda en memoria (no `localStorage`) para reducir robo vía XSS.
- `prefers-reduced-motion` desactiva animaciones y transiciones.
