# ZomboidNormalMapper

Herramienta para transformar texturas isométricas de Project Zomboid en
texturas planas y generarles un mapa de normales, pensada para usarlas en
Blender.

## Versiones

| Versión | Carpeta | Qué hace |
|---|---|---|
| ZomboidAssetWrapper | `CPP/` | Warp de perspectiva para paredes (C++ / OpenCV) |
| ZomboidNormalMaper | `CPP/` | Warp + mapa de normales con trackbars (C++ / OpenCV) |
| ZomboidNormalMapper | `Python/` | Aplicación de escritorio con puntos arrastrables, vista de luz, presets y deshacer (PySide6 + OpenCV + NumPy) |

## Uso (versión Python)

    pip install -r requirements.txt
    python ZomboidNormalMapper.py

## Créditos y Aviso Legal

Gracias a The Indie Stone por crear [Project Zomboid](https://projectzomboid.com),
lo cual hizo posible este proyecto.

Esta es una producción no oficial hecha por fans con fines no comerciales,
realizada bajo los [Términos y Condiciones de The Indie Stone](https://projectzomboid.com/blog/support/terms-conditions/).
Este proyecto no está afiliado ni respaldado por The Indie Stone.
