# Sign2Speech Design System (Apple-inspired Glassmorphism)

This design system acts as the source of truth for the Sign2Speech user interface, built according to the **TypeUI** specification standards. It defines color tokens, typography scales, spacing structures, bento grid configurations, PWA parameters, and interaction models.

---

## 1. Visual Brand & Foundations

*   **Theme**: Sleek Dark, Apple-inspired Glassmorphism with deep ambient radial gradients and frosted overlays.
*   **Contrast Philosophy**: High legibility matching WCAG 2.2 AA standards. Text should remain clear over dark or blurred surfaces.
*   **Tactile Feedback**: Subtle micro-vibrations (`navigator.vibrate`) for key interactive actions to reinforce feedback on mobile touchscreens.

---

## 2. Design Tokens

### Colors (Tailwind v4 Theme Definitions)

```css
@theme {
  --color-background: #050505;              /* Pure dark background base */
  --color-surface: #0a0a0a;                 /* Secondary layer surface */
  --color-surface-variant: #1c1c1e;         /* Elevated card background */
  
  --color-on-background: #f5f5f7;           /* Main readable body text (light slate) */
  --color-on-surface: #f5f5f7;              /* Headline and card title text */
  --color-on-surface-variant: #a1a1a6;      /* Secondary label and detail text */
  
  --color-primary: #b8c8df;                 /* Active/Accent (cool steel silver-blue) */
  --color-secondary: #c8c6c8;               /* Muted Accent (silver-gray) */
  --color-tertiary: #adc6ff;                /* Highlight accent */
  
  --color-outline: #8e9197;                 /* Medium border contrast */
  --color-outline-variant: #3a3a3c;         /* Low border contrast (fine divider lines) */
}
```

### Spacing & Layout
*   **Page Margin (Mobile)**: `16px` (`px-margin-mobile`)
*   **Page Margin (Desktop)**: `64px` (`px-margin-desktop`)
*   **Card Border Radius**: `1.25rem` (normal cards) to `2.5rem` (large bento cards/containers)
*   **Blur Filter**: `backdrop-blur-3xl` (heavy frosted look) to `backdrop-blur-md` (light HUD elements)

---

## 3. Layout Systems & Bento Guidelines

```
+-----------------------------------------------------------+
|                      Desktop Navbar                       |
+-----------------------------------------------------------+
|                                                           |
|                  Widescreen Video Panel                   |
|                        (Aspect 16:9)                      |
|                                                           |
+-----------------------------------------------------------+
|  [ Bento Card 1 ]    |  [ Bento Card 2 ]    |  [ Bento 3 ]|
|  Recognition Engine  |    NLP Refinement    |Session Data |
+-----------------------------------------------------------+
|                   Large Transcription Box                 |
+-----------------------------------------------------------+
|                  Floating Navigation Pill                 |
+-----------------------------------------------------------+
```

### Grid Spacing Rules
1.  **Desktop Bento Grids**: Use `grid grid-cols-1 md:grid-cols-3 gap-6` to distribute status widgets evenly.
2.  **Settings and History Panels**: Group logically into two columns (`grid grid-cols-1 md:grid-cols-2 gap-8`) to prevent horizontal stretching of content on widescreen viewports.
3.  **Floating Elements**: Bottom navigation bars and floating action buttons (FABs) must float with a `z-index` of `50` and heavy blurs (`apple-glass-dark`) to remain readable against scrolling content underneath.

---

## 4. Mobile & PWA Configuration (iOS & Android)

To guarantee native app performance when "downloaded" to the device, the following capabilities must be active:
*   **Viewport**: `<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no, viewport-fit=cover" />`
*   **Safari Capabilities**:
    *   `apple-mobile-web-app-capable`: `yes` (removes search bar/navigation chrome)
    *   `apple-mobile-web-app-status-bar-style`: `black-translucent` (blends page background into the iOS status bar)
    *   `apple-mobile-web-app-title`: `Sign2Speech`
*   **PWA Manifest**: Standardized `manifest.json` configured with standalone orientation, background colors matching `--color-background`, and maskable icons.

---

## 5. Interaction States & Transitions

*   **Interactive Hover**: Scale scale-98 or scale-102 with smooth durations (`duration-300 transition-all`).
*   **Active Touch**: Brief scale reduction (`active:scale-95`) to mimic button depression.
*   **Vibration Pattern**:
    *   *Micro feedback (toggles)*: `vibrate(10)`
    *   *Medium feedback (major actions)*: `vibrate(20)`
    *   *Heavy feedback (danger/reset)*: `vibrate(100)`

---

## 6. Prohibited Implementations (Anti-Patterns)

*   ❌ **Don't** use solid white backgrounds for panels; always use transparent overlays (`bg-white/5` or `apple-glass`) to preserve the dark ambient mesh glow.
*   ❌ **Don't** use hardcoded pixel widths for major layout blocks; rely on responsive Tailwind grids (`md:grid-cols-2`, `lg:grid-cols-3`) to support mobile stacked scaling.
*   ❌ **Don't** use low contrast text colors (like `#475569` on dark panels); fall back to `text-on-surface-variant` (`#a1a1a6`) for readability.
*   ❌ **Don't** hide the `<video>` element on camera toggle; keep it in the DOM with `opacity-0 pointer-events-none` to prevent React ref race conditions during startup.
