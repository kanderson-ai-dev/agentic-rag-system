# Frontend

Puro HTML/CSS/JS — sin framework, sin build step, sin dependencias externas.
Servido por FastAPI vía `StaticFiles` montado en `/` (ver `app/main.py`).

## Estructura

| Archivo      | Responsabilidad |
| ------------ | ------------------------------------------------------------------ |
| `index.html` | Marcado semántico + metadatos (SEO/OG/Twitter), logo y favicon SVG inline, script que aplica el tema antes del primer *paint* (anti-FOUC). |
| `styles.css` | Design system: tokens, reset, componentes base, temas claro/oscuro, layout responsive. |
| `app.js`     | Lógica: auth gate, chat, modal de revisión humana (HITL), dashboard, toggle de tema. |

## Branding y metadatos

- **Logo y favicon**: SVG inline (sin archivos binarios externos ni build step).
  El favicon se sirve como *data URI* (portable, sin request extra); el logo del
  header usa `currentColor`/tokens del design system para adaptarse al tema.
- **Metadatos**: `<title>` descriptivo, `<meta name="description">`,
  `theme-color` (variantes claro/oscuro vía `media`), y Open Graph + Twitter Card
  para que el enlace genere un preview rico al compartirse.
- **Estados del header**: el badge de auth tiene tres estados mutuamente
  excluyentes y visualmente distintos — *signed in*, *signed out* y
  *auth disabled* — gestionados por `setAuthStatus()` en `app.js`.

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
- **Tablas**: `table.recent` (envuelta en `.table-wrap` para scroll horizontal
  propio en pantallas estrechas, sin romper el scroll de la página).
- **Stats**: `.stat` (+ `.quality-stat.pass/.fail`).

## Experiencia de chat (Fase 4)

- **Markdown saneado (sin XSS)**: las respuestas del LLM se renderizan con un
  *renderer* propio (`renderMarkdown` en `app.js`) que construye nodos DOM solo
  con `textContent` / `createElement` — **nunca** asigna salida no confiable a
  `innerHTML`. El HTML crudo en la respuesta se trata siempre como texto literal
  (riesgo de XSS neutralizado por diseño). Subconjunto soportado: código fenced
  e inline, headings, listas ordenadas/desordenadas, blockquotes, párrafos y
  `**negrita**` / `*cursiva*`.
- **Indicador "escribiendo…"** y estados de carga: burbuja con puntos animados
  (`showTyping`/`hideTyping`) mientras la query está en vuelo; el botón *Send*
  pasa a "*Sending…*" y se deshabilita hasta resolver.
- **Copiar respuesta y fuentes**: cada respuesta trae un botón *Copy* (Clipboard
  API con fallback a `execCommand("copy")`) y las fuentes recuperadas se
  muestran como *chips* (`sources` del `QueryResponse`).
- **Estados**: *welcome* (vacío), *loading* (typing), y *error* con reintento
  (`addErrorWithRetry`).
- **Auto-scroll inteligente** (solo si ya estás cerca del fondo) y **textarea
  adaptativo** que crece con el contenido hasta un máximo.

## Layout responsive (mobile-first)

El layout es **mobile-first**: los estilos base apuntan a la pantalla más pequeña
(móvil 320px) y se mejoran progresivamente con `min-width` media queries. No hay
scroll horizontal en ningún tamaño.

| Breakpoint | Token CSS               | Qué cambia |
| ---------- | ----------------------- | ---------------------------------------------------------- |
| Móvil      | (base)                  | `.layout` a una columna, gutters compactos, header sticky con título truncado (ellipsis). |
| Tablet     | `--bp-tablet` (≥640px)  | Gutters y padding de cards más amplios; padding del header. |
| Escritorio | `--bp-desktop` (≥1024px)| Formulario de login acotado, stats en más columnas. |

Convenciones de jerarquía y espaciado:

- Todo el contenido vive en `.layout` (CSS Grid, `minmax(0, 1fr)` para evitar
  desbordes), alineado y centrado con `max-width: var(--max-width)`.
- El header es `position: sticky` con `z-index` sobre el contenido.
- Las tablas usan `.table-wrap` (`overflow-x: auto`) en vez de forzar scroll de
  página en móvil.

## Convenciones

- El contenido dinámico se inserta con `textContent` (o `innerHTML` solo para
  estructura estática conocida); **nunca** se renderiza salida del LLM sin sanear.
- El JWT se guarda en memoria (no `localStorage`) para reducir robo vía XSS.
- `prefers-reduced-motion` desactiva animaciones y transiciones.
