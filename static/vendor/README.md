# Vendored third-party scripts

| File | Version | SHA-384 |
|---|---|---|
| `htmx.min.js` | htmx.org 2.0.4 (`dist/htmx.min.js`) | `1c67f3b687e8b5fb21705efef27e382502f6a099a8c150a13d3838f12fa3bd9a33b4f7efc03efd382fcb4ed1ba74ca7e` |

Vendored instead of loaded from unpkg so the app works offline and on a LAN
without a CDN, and so a CDN compromise cannot inject code (OWASP A08). The
hash was computed on 2026-10-02 from the unpkg and jsdelivr copies, which were
byte-identical. To upgrade: download the new `dist/htmx.min.js`, verify the
hash against a second mirror, replace the file and update this table.
