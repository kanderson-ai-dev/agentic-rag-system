# Frontend

Puro HTML/CSS/JS — sin framework, sin build step, sin dependencias externas.
Servido por FastAPI vía `StaticFiles` montado en `/` (ver `app/main.py`).

## Estructura

| Archivo           | Responsabilidad |
| ----------------- | ------------------------------------------------------------------ |
| `index.html`      | Marcado semántico + metadatos (SEO/OG/Twitter), logo y favicon SVG inline, script que aplica el tema antes del primer *paint* (anti-FOUC). |
| `styles.css`      | Design system: tokens, reset, componentes base, temas claro/oscuro, layout responsive. |
| `js/app.js`       | **Entrypoint** (ES module). Orquesta los módulos y cablea dependencias cruzadas. Único módulo con efectos laterales al importarse. No hay auth gate: el chat y el dashboard están disponibles de inmediato. |
| `js/util.js`      | Helpers puros sin efectos: `$`, `$$`, `el`, `getSessionId`, `debounce`. |
| `js/theme.js`     | Toggle de tema claro/oscuro y persistencia en `localStorage`. |
| `js/toast.js`     | Notificaciones transitorias (`showToast`), anuncios `aria-live` (`announce`) y manejo global de errores. |
| `js/markdown.js`  | Renderer de Markdown **saneado** (sin XSS), construye DOM con `textContent`/`createElement`. |
| `js/api.js`       | Wrapper `fetch` + `X-Session-Id`. Sin flujo de login: la UI nunca envía credenciales. |
| `js/chat.js`      | Experiencia de chat: mensajes, typing, copiar, composer adaptativo, envío de query. |
| `js/review.js`    | Modal de revisión humana (HITL). |
| `js/dashboard.js` | Dashboard: métricas, tabla ordenable/paginada, gráfico SVG y Quality (EDD). |

### Modularización (Fase 8)

El frontend usa **módulos ES nativos** (`<script type="module">`), sin build step
ni bundler — el navegador los carga con `import`/`export` relativos. Ventajas:

- **Sin estado global innecesario**: el `thread_id`, el estado del composer y
  del modal viven en el *scope* de su módulo (no en `window`), y se comparten
  entre módulos mediante *accessors* explícitos (`getThreadId`, …) o
  *callbacks inyectados* (`wireChat`, `wireReview`), evitando imports
  circulares.
- **Dependencias unidireccionales**: `app.js` es el único que conoce el grafo de
  inicialización; cada módulo expone un `init*` idempotente.
- **`debounce` en inputs**: los manejadores de alta frecuencia (`resize` que
  redibuja el gráfico SVG) se desacoplan con `debounce` desde `util.js`.
- **Sin *layout shift***: se reserva espacio vertical (`min-height`) en las
  tarjetas de métricas, la región del gráfico y la tabla de requests recientes,
  para que las transiciones *skeleton → datos → vacío* no desplacen el contenido.

## Branding y metadatos

- **Logo y favicon**: SVG inline (sin archivos binarios externos ni build step).
  El favicon se sirve como *data URI* (portable, sin request extra); el logo del
  header usa `currentColor`/tokens del design system para adaptarse al tema.
- **Metadatos**: `<title>` descriptivo, `<meta name="description">`,
  `theme-color` (variantes claro/oscuro vía `media`), y Open Graph + Twitter Card
  para que el enlace genere un preview rico al compartirse.

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
- **Badges**: `.badge`.
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

## Flujo de revisión humana — HITL (Fase 5)

Cuando el bucle de corrección Self-RAG agota sus reintentos, el grafo se pausa
(`interrupt()`) y el frontend abre un **modal accesible** (`<dialog>`) para que
un humano decida cómo continuar. Todo el flujo es operable solo con teclado.

- **Modal accesible**: `<dialog>` nativo con `aria-labelledby` y
  `aria-describedby`, **foco atrapado** y **cierre con `Esc`** (comportamiento
  nativo del elemento). Al abrirse, el foco se mueve al primer control de
  decisión.
- **Tres opciones explicadas**: cada decisión es un *radio input* bajo un
  `<fieldset>` (navegable con flechas), con su explicación visible:
  - **Approve** — acepta la mejor respuesta disponible tal cual.
  - **Retry** — re-ejecuta el retrieval con una pregunta revisada que tú
    escribes.
  - **Override** — escribes manualmente la respuesta correcta.
- **Validación del input**: `retry`/`override` revelan un campo etiquetado; si
  se envía vacío, se bloquea el envío y se muestra un error visible con foco
  sobre el campo (se limpia al volver a escribir). Si no se eligió ninguna
  opción, se pide elegir.
- **Estados de carga/error**: al confirmar, el botón pasa a "*Submitting…*" y
  se deshabilitan los controles; si la petición falla, el error se muestra
  *dentro* del modal (`role="alert"`) y el modal permanece abierto para
  corregir y reintentar — no se cierra ni se vuelca el error al chat.

## Dashboard (Fase 6)

El panel `#dashboard-panel` muestra datos reales del backend
(`/api/v1/dashboard/summary`, `/recent` y `/quality`), todos escritos con
`textContent` (nunca `innerHTML` con datos del servidor):

- **Tarjetas de métricas**: costo total (`$`), latencia promedio (`ms`),
  requests bloqueados por el guardrail y escalados a revisión humana
  (`renderSummary`).
- **Quality (EDD)**: indicadores pass/fail por umbral del *scorecard* RAGAS
  (`renderQuality`), con estado implícito "sin scorecard todavía".
- **Gráfico de costo/latencia**: SVG inline construido con `createElementNS`
  (sin librerías ni `<canvas>`) con dos series normalizadas — *latency* (acento)
  y *cost* (éxito) — y una leyenda con el valor pico real de cada una
  (`renderChart`).
- **Tabla de requests recientes**: columnas ordenables por teclado (botones
  `button.th-sort` con `aria-sort` en el `<th>`), paginación client-side
  (`renderPagination`), y formatos `$`/`ms`/miles (`formatUsd`,
  `formatLatency`, `formatTokens`).
- **Estados vacío y cargando**: skeleton con *shimmer* mientras los datos están
  en vuelo (`renderDashboardSkeleton`) y mensajes de vacío explícitos cuando no
  hay actividad (`#recent-empty`, `#chart-empty`).

## Layout responsive (mobile-first)

El layout es **mobile-first**: los estilos base apuntan a la pantalla más pequeña
(móvil 320px) y se mejoran progresivamente con `min-width` media queries. No hay
scroll horizontal en ningún tamaño.

| Breakpoint | Token CSS               | Qué cambia |
| ---------- | ----------------------- | ---------------------------------------------------------- |
| Móvil      | (base)                  | `.layout` a una columna, gutters compactos, header sticky con título truncado (ellipsis). |
| Tablet     | `--bp-tablet` (≥640px)  | Gutters y padding de cards más amplios; padding del header. |
| Escritorio | `--bp-desktop` (≥1024px)| Stats en más columnas. |

Convenciones de jerarquía y espaciado:

- Todo el contenido vive en `.layout` (CSS Grid, `minmax(0, 1fr)` para evitar
  desbordes), alineado y centrado con `max-width: var(--max-width)`.
- El header es `position: sticky` con `z-index` sobre el contenido.
- Las tablas usan `.table-wrap` (`overflow-x: auto`) en vez de forzar scroll de
  página en móvil.

## Convenciones

- El contenido dinámico se inserta con `textContent` (o `innerHTML` solo para
  estructura estática conocida); **nunca** se renderiza salida del LLM sin sanear.
- `prefers-reduced-motion` desactiva animaciones y transiciones.

## Accesibilidad y seguridad (Fase 7)

- **Anuncios en vivo (`aria-live`)**: la conversación es un `role="log"` polito y,
  además, un *announcer* visualmente oculto (`#chat-announcer`, `aria-live="polite"`)
  anuncia en voz alta eventos clave — "*Answer received*", "*A human review is
  required*" y "*Request blocked by the input guardrail*" — sin mover el foco.
- **Navegación por teclado completa**: *skip link* ("Skip to main content") que
  salta al `<main id="main">` (con `tabindex="-1"`), y gestión explícita de foco
  en el modal: al abrirse, el foco se mueve al primer *radio* de decisión y, al
  cerrarse, se restaura al elemento que lo tenía antes (`lastFocusedElement`).
- **Contraste ≥ AA y `prefers-reduced-motion`**: paleta verificada contra el tema
  activo; el bloque `@media (prefers-reduced-motion: reduce)` colapsa todas las
  animaciones/transiciones (WCAG 2.3.3).
- **Saneado total**: los módulos ES (`js/*.js`) **nunca** asignan a `innerHTML` —
  todo el contenido dinámico (respuesta del LLM, dashboard, errores) usa
  `textContent`/`createElement`. No existe ningún camino de código que traduzca un
  payload XSS en markup ejecutable.
- **Toasts y manejo global de errores**: un contenedor `#toasts` (`role="status"`,
  `aria-live="polite"`) muestra notificaciones transitorias descartables
  (`showToast`), y los listeners globales `window.addEventListener("error")` y
  `("unhandledrejection")` convierten fallos inesperados en una notificación
  visible en lugar de morir en silencio en la consola.
