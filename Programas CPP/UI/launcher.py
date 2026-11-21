"""
launcher.py
Interfaz de interacción con los temas vistos en Visión Artificial.

## Idea general
Los programas del repositorio están escritos en C++ con OpenCV y cada uno abre sus
propias ventanas. Esta interfaz NO los reescribe: es un lanzador. Para el programa que
elijas en el árbol de temas:

  1. Muestra su descripción, los controles de sus ventanas y un formulario con los
     datos que necesita (imagen, tamaño de kernel, parche, etc.).
  2. Lo COMPILA con g++ + pkg-config (solo si el ejecutable no existe o el código
     cambió) dejando los binarios en UI/build/.
  3. Lo EJECUTA en una carpeta de salidas propia (UI/salidas/<programa>/), entregándole
     los datos del formulario como argumentos o por stdin, tal como si los escribieras
     en la consola. Sus ventanas de OpenCV se abren aparte y se cierran con q / ESC.
  4. Muestra la consola del programa, las imágenes/archivos que generó (pestaña
     "Resultados") y su código fuente con colores (pestaña "Código").

Cómo se describe cada programa (campos, argumentos, stdin) está en programas.py.

## Requisitos
  - Python 3.9+ y PySide6:    pip install -r UI/requirements.txt
  - Compilador C++ (g++) y OpenCV 4 visibles con pkg-config (ver README del repositorio).
    Si tu entorno es distinto (p. ej. Visual Studio) hay dos salidas desde la interfaz:
      * "Configuración del compilador..." para escribir tu propio comando, o
      * "Elegir..." para usar un .exe que ya compilaste tú.

## Ejecutar
  python UI/launcher.py
"""
from __future__ import annotations

import html
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QProcess, QSettings, Qt, QUrl
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QFontDatabase, QPalette,
                           QPixmap, QSyntaxHighlighter, QTextCharFormat, QTextCursor)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
                               QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QSpinBox, QSplitter,
                               QStackedWidget, QTabWidget, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from programas import PROGRAMAS, RAIZ, Programa

AQUI = Path(__file__).resolve().parent
BUILD = AQUI / "build"       # ejecutables compilados
SALIDAS = AQUI / "salidas"   # carpeta de trabajo de cada programa (imágenes generadas, .yml...)
EXT_BINARIO = ".exe" if os.name == "nt" else ""
EXT_IMAGEN = {".png", ".jpg", ".jpeg", ".bmp"}
EXT_TEXTO = {".yml", ".yaml", ".txt", ".csv", ".json"}

# Colores de la interfaz.
C_FONDO, C_PANEL, C_TEXTO = "#1e1f24", "#2a2c33", "#e6e6e6"
C_ACENTO, C_OK, C_AVISO, C_ERROR, C_APAGADO = "#4f8cff", "#5ec27a", "#e0a73a", "#ef6b6b", "#8d93a0"


# ==========================================================================
# Resaltado de sintaxis sencillo (C++ y Python) para la pestaña "Código"
# ==========================================================================
class Resaltador(QSyntaxHighlighter):
    PALABRAS_CPP = (r"\b(int|float|double|bool|char|void|auto|const|static|return|if|else|for|while|"
                    r"switch|case|break|continue|struct|class|using|namespace|include|true|false|"
                    r"string|vector|nullptr|unsigned|long|size_t|uchar|define|ifdef|endif|if)\b")
    PALABRAS_PY = (r"\b(def|class|import|from|return|if|elif|else|for|while|in|not|and|or|None|True|"
                   r"False|with|as|try|except|finally|raise|lambda|self|pass|break|continue)\b")

    def __init__(self, documento, python: bool):
        super().__init__(documento)
        self.python = python

        def fmt(color, cursiva=False, negrita=False):
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            f.setFontItalic(cursiva)
            if negrita:
                f.setFontWeight(QFont.Bold)
            return f

        self.f_palabra = fmt("#c586c0", negrita=True)
        self.f_numero = fmt("#b5cea8")
        self.f_cadena = fmt("#ce9178")
        self.f_comentario = fmt("#7f8a99", cursiva=True)
        self.re_palabra = re.compile(self.PALABRAS_PY if python else self.PALABRAS_CPP)
        self.re_numero = re.compile(r"\b\d+(\.\d+)?f?\b")
        self.re_cadena = re.compile(r'"(\\.|[^"\\])*"')
        self.re_linea = re.compile(r"#.*" if python else r"//.*")

    def highlightBlock(self, texto):
        for rx, f in ((self.re_palabra, self.f_palabra), (self.re_numero, self.f_numero),
                      (self.re_cadena, self.f_cadena), (self.re_linea, self.f_comentario)):
            for m in rx.finditer(texto):
                self.setFormat(m.start(), m.end() - m.start(), f)
        if self.python:
            return
        # Comentarios de bloque /* ... */ (los programas los usan en la cabecera).
        # Estado 1 = dentro de un comentario que continúa en la línea siguiente.
        self.setCurrentBlockState(0)
        inicio = 0 if self.previousBlockState() == 1 else texto.find("/*")
        while inicio >= 0:
            fin = texto.find("*/", inicio if self.previousBlockState() != 1 else 0)
            if fin == -1:
                self.setCurrentBlockState(1)
                largo = len(texto) - inicio
            else:
                largo = fin - inicio + 2
            self.setFormat(inicio, largo, self.f_comentario)
            inicio = texto.find("/*", inicio + largo)


# ==========================================================================
# Diálogo: comando de compilación personalizado
# ==========================================================================
class DialogoCompilador(QDialog):
    def __init__(self, padre, plantilla: str, por_defecto: str):
        super().__init__(padre)
        self.setWindowTitle("Configuración del compilador")
        self.resize(640, 260)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "Comando de compilación. Usa {src} para el .cpp y {out} para el ejecutable.\n"
            "Déjalo vacío para usar el automático (g++ + pkg-config opencv4)."))
        self.edit = QLineEdit(plantilla)
        self.edit.setPlaceholderText('g++ -std=c++17 "{src}" -o "{out}" <flags de OpenCV>')
        lay.addWidget(self.edit)
        ayuda = QLabel("Automático detectado:\n" + por_defecto)
        ayuda.setWordWrap(True)
        ayuda.setStyleSheet(f"color:{C_APAGADO}")
        lay.addWidget(ayuda)
        lay.addStretch()
        fila = QHBoxLayout()
        fila.addStretch()
        for texto, accion in (("Cancelar", self.reject), ("Guardar", self.accept)):
            b = QPushButton(texto)
            b.clicked.connect(accion)
            fila.addWidget(b)
        lay.addLayout(fila)

    def plantilla(self) -> str:
        return self.edit.text().strip()


# ==========================================================================
# Ventana principal
# ==========================================================================
class Ventana(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visión Artificial — Interfaz de interacción")
        self.resize(1280, 820)
        self.ajustes = QSettings("AV-Playground", "Launcher")

        self.actual: Programa | None = None
        self.getters: dict = {}            # nombre de campo -> función que lee su valor
        self.guardados: dict = {}          # id programa -> valores del formulario
        self.proceso: QProcess | None = None
        self.compilando = False
        self.t_inicio = 0.0
        self.carpeta_resultados: Path | None = None

        self._construir_ui()
        self._llenar_arbol()

    # ---------------- construcción de la interfaz ----------------
    def _construir_ui(self):
        raiz = QSplitter(Qt.Horizontal)
        self.setCentralWidget(raiz)

        # Izquierda: árbol de temas
        izq = QWidget()
        li = QVBoxLayout(izq)
        li.setContentsMargins(12, 12, 6, 12)
        titulo = QLabel("Temas vistos")
        titulo.setStyleSheet("font-size:16px; font-weight:600;")
        li.addWidget(titulo)
        self.arbol = QTreeWidget()
        self.arbol.setHeaderHidden(True)
        self.arbol.setIndentation(14)
        self.arbol.currentItemChanged.connect(self._al_elegir)
        li.addWidget(self.arbol, 1)
        b_comp = QPushButton("Configuración del compilador…")
        b_comp.clicked.connect(self._config_compilador)
        li.addWidget(b_comp)
        raiz.addWidget(izq)

        # Derecha: detalle del programa
        der = QWidget()
        ld = QVBoxLayout(der)
        ld.setContentsMargins(6, 12, 12, 12)

        self.l_titulo = QLabel()
        self.l_titulo.setStyleSheet("font-size:22px; font-weight:600;")
        self.l_grupo = QLabel()
        self.l_grupo.setStyleSheet(f"color:{C_ACENTO}")
        self.l_desc = QLabel()
        self.l_desc.setWordWrap(True)
        self.l_aviso = QLabel()
        self.l_aviso.setWordWrap(True)
        self.l_aviso.setStyleSheet(
            f"color:{C_AVISO}; background:{C_PANEL}; border-left:3px solid {C_AVISO}; padding:6px 8px;")
        self.l_teclas = QLabel()
        self.l_teclas.setWordWrap(True)
        self.l_teclas.setStyleSheet(f"color:{C_APAGADO}")
        for w in (self.l_titulo, self.l_grupo, self.l_desc, self.l_aviso, self.l_teclas):
            ld.addWidget(w)

        # Formulario + vista previa de la imagen elegida
        caja = QGroupBox("Parámetros")
        fila_caja = QHBoxLayout(caja)
        self.form = QFormLayout()
        self.form.setLabelAlignment(Qt.AlignRight)
        fila_caja.addLayout(self.form, 1)
        self.previa = QLabel()
        self.previa.setFixedSize(220, 150)
        self.previa.setAlignment(Qt.AlignCenter)
        self.previa.setStyleSheet(f"background:{C_PANEL}; border-radius:6px; color:{C_APAGADO}")
        fila_caja.addWidget(self.previa)
        ld.addWidget(caja)

        # Botones
        fila = QHBoxLayout()
        self.b_ejecutar = QPushButton("▶  Ejecutar")
        self.b_ejecutar.setObjectName("primario")
        self.b_ejecutar.clicked.connect(self.ejecutar)
        self.b_compilar = QPushButton("Compilar")
        self.b_compilar.clicked.connect(lambda: self.compilar(self.actual, None))
        self.b_detener = QPushButton("■  Detener")
        self.b_detener.clicked.connect(self.detener)
        self.b_salidas = QPushButton("Abrir carpeta de salidas")
        self.b_salidas.clicked.connect(self._abrir_salidas)
        for b in (self.b_ejecutar, self.b_compilar, self.b_detener):
            fila.addWidget(b)
        fila.addStretch()
        fila.addWidget(self.b_salidas)
        ld.addLayout(fila)

        # Ejecutable propio (alternativa a compilar desde la interfaz)
        fila_exe = QHBoxLayout()
        self.l_exe = QLabel()
        self.l_exe.setStyleSheet(f"color:{C_APAGADO}")
        self.b_exe = QPushButton("Elegir…")
        self.b_exe.clicked.connect(self._elegir_exe)
        self.b_exe_reset = QPushButton("Restablecer")
        self.b_exe_reset.clicked.connect(self._restablecer_exe)
        fila_exe.addWidget(self.l_exe, 1)
        fila_exe.addWidget(self.b_exe)
        fila_exe.addWidget(self.b_exe_reset)
        ld.addLayout(fila_exe)

        # Pestañas: consola / resultados / código
        self.pestanas = QTabWidget()
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        mono.setPointSize(10)

        self.consola = QPlainTextEdit()
        self.consola.setReadOnly(True)
        self.consola.setFont(mono)
        self.pestanas.addTab(self.consola, "Consola")

        res = QWidget()
        lr = QHBoxLayout(res)
        self.lista_res = QListWidget()
        self.lista_res.setMaximumWidth(240)
        self.lista_res.currentRowChanged.connect(self._mostrar_resultado)
        lr.addWidget(self.lista_res)
        self.pila_res = QStackedWidget()
        self.img_res = QLabel("Aún no hay resultados.\nEjecuta el programa para ver lo que genera.")
        self.img_res.setAlignment(Qt.AlignCenter)
        self.img_res.setStyleSheet(f"color:{C_APAGADO}")
        self.txt_res = QPlainTextEdit()
        self.txt_res.setReadOnly(True)
        self.txt_res.setFont(mono)
        self.pila_res.addWidget(self.img_res)
        self.pila_res.addWidget(self.txt_res)
        lr.addWidget(self.pila_res, 1)
        self.pestanas.addTab(res, "Resultados")
        self.pixmap_res: QPixmap | None = None

        self.codigo = QPlainTextEdit()
        self.codigo.setReadOnly(True)
        self.codigo.setFont(mono)
        self.codigo.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.resaltador = None
        self.pestanas.addTab(self.codigo, "Código fuente")

        ld.addWidget(self.pestanas, 1)
        raiz.addWidget(der)
        raiz.setStretchFactor(0, 0)
        raiz.setStretchFactor(1, 1)
        raiz.setSizes([300, 980])
        self.statusBar().showMessage("Elige un tema en el árbol de la izquierda.")

    def _llenar_arbol(self):
        grupos: dict = {}
        for p in PROGRAMAS:
            if p.grupo not in grupos:
                g = QTreeWidgetItem(self.arbol, [p.grupo])
                g.setFlags(Qt.ItemIsEnabled)
                f = g.font(0)
                f.setBold(True)
                g.setFont(0, f)
                grupos[p.grupo] = g
            hijo = QTreeWidgetItem(grupos[p.grupo], [p.titulo])
            hijo.setData(0, Qt.UserRole, p.id)
        self.arbol.expandAll()
        primero = next(iter(grupos.values())).child(0)
        self.arbol.setCurrentItem(primero)

    # ---------------- selección de programa ----------------
    def _al_elegir(self, item, _previo):
        if item is None or item.data(0, Qt.UserRole) is None:
            return
        if self.actual is not None:
            self.guardados[self.actual.id] = self._leer_valores()
        p = next(x for x in PROGRAMAS if x.id == item.data(0, Qt.UserRole))
        self.actual = p
        self.l_titulo.setText(p.titulo)
        self.l_grupo.setText(p.grupo)
        self.l_desc.setText(p.descripcion)
        self.l_aviso.setText("⚠  " + p.aviso)
        self.l_aviso.setVisible(bool(p.aviso))
        self.l_teclas.setText("Teclas: " + p.teclas if p.teclas else "")
        self.l_teclas.setVisible(bool(p.teclas))
        self._construir_formulario(p)
        self._cargar_codigo(p)
        self._actualizar_exe()
        self._limpiar_resultados()
        self.b_compilar.setVisible(p.tipo == "cpp")
        for w in (self.l_exe, self.b_exe, self.b_exe_reset):
            w.setVisible(p.tipo == "cpp")
        self.statusBar().showMessage(f"{p.titulo}: listo.")

    def _construir_formulario(self, p: Programa):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.getters = {}
        previos = self.guardados.get(p.id, {})
        campo_imagen = None
        for c in p.campos:
            valor = previos.get(c.nombre, c.default)
            if c.tipo in ("imagen", "carpeta"):
                edit = QLineEdit(str(valor or ""))
                btn = QPushButton("Examinar…")
                btn.clicked.connect(lambda _=False, e=edit, cc=c: self._examinar(e, cc))
                fila = QHBoxLayout()
                fila.addWidget(edit, 1)
                fila.addWidget(btn)
                cont = QWidget()
                cont.setLayout(fila)
                fila.setContentsMargins(0, 0, 0, 0)
                self.form.addRow(c.etiqueta + ":", cont)
                self.getters[c.nombre] = edit.text
                if c.tipo == "imagen":
                    campo_imagen = edit
            elif c.tipo == "int":
                sp = QSpinBox()
                sp.setRange(c.minimo, c.maximo)
                sp.setValue(int(valor))
                self.form.addRow(c.etiqueta + ":", sp)
                self.getters[c.nombre] = sp.value
            elif c.tipo == "bool":
                cb = QCheckBox(c.etiqueta)
                cb.setChecked(bool(valor))
                self.form.addRow("", cb)
                self.getters[c.nombre] = cb.isChecked
            elif c.tipo == "opcion":
                co = QComboBox()
                co.addItems(c.opciones)
                co.setCurrentText(str(valor))
                self.form.addRow(c.etiqueta + ":", co)
                self.getters[c.nombre] = co.currentText
            if c.ayuda:
                ayuda = QLabel(c.ayuda)
                ayuda.setStyleSheet(f"color:{C_APAGADO}; font-size:11px")
                self.form.addRow("", ayuda)
        self.previa.setVisible(campo_imagen is not None)
        if campo_imagen is not None:
            campo_imagen.textChanged.connect(self._actualizar_previa)
            self._actualizar_previa(campo_imagen.text())

    def _examinar(self, edit: QLineEdit, c):
        inicio = edit.text() or str(RAIZ)
        if c.tipo == "carpeta":
            ruta = QFileDialog.getExistingDirectory(self, c.etiqueta, inicio)
        else:
            ruta, _ = QFileDialog.getOpenFileName(
                self, c.etiqueta, inicio, "Imágenes (*.png *.jpg *.jpeg *.bmp *.tif);;Todos (*)")
        if ruta:
            edit.setText(ruta)

    def _actualizar_previa(self, ruta: str):
        pm = QPixmap(ruta.strip().strip('"')) if ruta.strip() else QPixmap()
        if pm.isNull():
            self.previa.setPixmap(QPixmap())
            self.previa.setText("Sin vista previa")
        else:
            self.previa.setPixmap(pm.scaled(self.previa.size(), Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation))

    def _leer_valores(self) -> dict:
        return {nombre: getter() for nombre, getter in self.getters.items()}

    def _validar(self, p: Programa, v: dict) -> str:
        for c in p.campos:
            val = str(v.get(c.nombre) or "").strip().strip('"')
            if c.tipo == "imagen":
                # Con "webcam" marcada (morfología) no hace falta imagen.
                if v.get("webcam") or (c.opcional and not val):
                    continue
                if not Path(val).is_file():
                    return f"{c.etiqueta}: el archivo no existe.\n{val}"
            elif c.tipo == "carpeta" and not Path(val).is_dir():
                return f"{c.etiqueta}: la carpeta no existe.\n{val}"
        return ""

    # ---------------- código fuente ----------------
    def _cargar_codigo(self, p: Programa):
        try:
            texto = p.fuente.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            texto = f"No se pudo leer {p.fuente}: {e}"
        self.codigo.setPlainText(texto)
        self.resaltador = Resaltador(self.codigo.document(), p.tipo == "python")

    # ---------------- compilación ----------------
    def _ejecutable_propio(self, p: Programa) -> str:
        ruta = self.ajustes.value(f"exe/{p.id}", "")
        return ruta if ruta and Path(ruta).is_file() else ""

    def _ejecutable(self, p: Programa) -> Path:
        propio = self._ejecutable_propio(p)
        return Path(propio) if propio else BUILD / (p.id + EXT_BINARIO)

    def _actualizar_exe(self):
        p = self.actual
        propio = self._ejecutable_propio(p)
        if propio:
            self.l_exe.setText(f"Ejecutable propio: {propio}")
        else:
            exe = BUILD / (p.id + EXT_BINARIO)
            estado = "compilado" if exe.exists() else "se compilará al ejecutar"
            self.l_exe.setText(f"Ejecutable: {estado} ({exe.name})")
        self.b_exe_reset.setEnabled(bool(propio))

    def _elegir_exe(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Ejecutable ya compilado", str(RAIZ))
        if ruta:
            self.ajustes.setValue(f"exe/{self.actual.id}", ruta)
            self._actualizar_exe()

    def _restablecer_exe(self):
        self.ajustes.remove(f"exe/{self.actual.id}")
        self._actualizar_exe()

    def _flags_opencv(self) -> list:
        """Flags de compilación/enlace de OpenCV vía pkg-config (opencv4 u opencv)."""
        if not shutil.which("pkg-config"):
            raise RuntimeError("No se encontró 'pkg-config'. Instálalo o escribe tu propio comando "
                               "en 'Configuración del compilador…'.")
        for nombre in ("opencv4", "opencv"):
            r = subprocess.run(["pkg-config", "--cflags", "--libs", nombre],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return shlex.split(r.stdout)
        raise RuntimeError("pkg-config no encuentra OpenCV (opencv4). Revisa PKG_CONFIG_PATH.")

    def _comando_compilacion(self, p: Programa, salida: Path) -> list:
        plantilla = self.ajustes.value("compilador/plantilla", "")
        if plantilla:
            cmd = plantilla.format(src=str(p.fuente), out=str(salida))
            return shlex.split(cmd, posix=(os.name != "nt"))
        cxx = next((c for c in ("g++", "clang++", "c++") if shutil.which(c)), None)
        if not cxx:
            raise RuntimeError("No se encontró un compilador C++ (g++/clang++).")
        return [cxx, "-std=c++17", str(p.fuente), "-o", str(salida)] + self._flags_opencv()

    def _config_compilador(self):
        try:
            defecto = " ".join(self._comando_compilacion(self.actual or PROGRAMAS[0], Path("<salida>")))
        except RuntimeError as e:
            defecto = str(e)
        d = DialogoCompilador(self, self.ajustes.value("compilador/plantilla", ""), defecto)
        if d.exec() == QDialog.Accepted:
            self.ajustes.setValue("compilador/plantilla", d.plantilla())
            self.statusBar().showMessage("Configuración del compilador guardada.")

    def _necesita_compilar(self, p: Programa) -> bool:
        if self._ejecutable_propio(p):
            return False
        exe = BUILD / (p.id + EXT_BINARIO)
        return not exe.exists() or p.fuente.stat().st_mtime > exe.stat().st_mtime

    def compilar(self, p: Programa, despues):
        """Compila p; si tiene éxito llama a 'despues()' (si se dio)."""
        if p.tipo != "cpp" or self.proceso is not None:
            return
        BUILD.mkdir(exist_ok=True)
        salida = BUILD / (p.id + EXT_BINARIO)
        try:
            cmd = self._comando_compilacion(p, salida)
        except RuntimeError as e:
            self._error("No se puede compilar", str(e))
            return
        self.pestanas.setCurrentIndex(0)
        # Se muestra resumido: los flags de OpenCV son decenas de -lopencv_* y no aportan.
        visibles = [c for c in cmd if not c.startswith(("-l", "-L", "-I", "-Wl"))]
        self._log(f"\n$ {' '.join(shlex.quote(c) for c in visibles)} … (+ flags de OpenCV)\n", C_APAGADO)
        self.compilando = True
        self._ocupado(True)
        self.statusBar().showMessage(f"Compilando {p.fuente.name}…")
        self.proceso = QProcess(self)
        self.proceso.readyReadStandardOutput.connect(lambda: self._leer(self.proceso, False))
        self.proceso.readyReadStandardError.connect(lambda: self._leer(self.proceso, True))
        self.proceso.finished.connect(lambda code, _s: self._compilacion_terminada(code, p, despues))
        self.proceso.errorOccurred.connect(self._proceso_error)
        self.proceso.start(cmd[0], cmd[1:])

    def _compilacion_terminada(self, code: int, p: Programa, despues):
        self.proceso = None
        self.compilando = False
        self._ocupado(False)
        self._actualizar_exe()
        if code == 0:
            self._log("✔ Compilación correcta.\n", C_OK)
            self.statusBar().showMessage("Compilación correcta.")
            if despues:
                despues()
        else:
            self._log(f"✘ La compilación falló (código {code}).\n", C_ERROR)
            self.statusBar().showMessage("La compilación falló. Revisa la consola.")

    # ---------------- ejecución ----------------
    def ejecutar(self):
        p = self.actual
        if p is None or self.proceso is not None:
            return
        v = self._leer_valores()
        for k, val in v.items():
            if isinstance(val, str):
                v[k] = val.strip().strip('"')
        err = self._validar(p, v)
        if err:
            self._error("Revisa los parámetros", err)
            return
        if p.tipo == "cpp" and self._necesita_compilar(p):
            self.compilar(p, lambda: self._lanzar(p, v))
        else:
            self._lanzar(p, v)

    def _lanzar(self, p: Programa, v: dict):
        trabajo = SALIDAS / p.id
        trabajo.mkdir(parents=True, exist_ok=True)
        try:
            if p.preparar:
                p.preparar(trabajo, v)
            args, entrada = p.construir(v)
        except (ValueError, OSError) as e:
            self._error("No se pudo preparar la ejecución", str(e))
            return

        if p.tipo == "python":
            programa, argumentos, cwd = sys.executable, ["-u", str(p.fuente)] + args, p.fuente.parent
        else:
            exe = self._ejecutable(p)
            programa, argumentos, cwd = str(exe), args, trabajo
            # La salida de C++ por una tubería queda en búfer hasta el final; con stdbuf
            # (Linux/macOS con coreutils) se obtiene línea a línea.
            if shutil.which("stdbuf"):
                programa, argumentos = "stdbuf", ["-oL", "-eL", str(exe)] + args

        self.pestanas.setCurrentIndex(0)
        self._log(f"\n▶ {p.titulo}\n", C_ACENTO)
        self._log(f"$ {' '.join(shlex.quote(a) for a in [programa] + argumentos)}\n", C_APAGADO)
        if entrada.strip():
            self._log("(stdin) " + entrada.strip().replace("\n", " ⏎ ") + "\n", C_APAGADO)
        self.t_inicio = time.time()
        self.carpeta_resultados = trabajo if p.muestra_resultados else None
        self._ocupado(True)
        self.statusBar().showMessage(f"Ejecutando {p.titulo}… cierra sus ventanas con q / ESC.")

        self.proceso = QProcess(self)
        self.proceso.setWorkingDirectory(str(cwd))
        self.proceso.readyReadStandardOutput.connect(lambda: self._leer(self.proceso, False))
        self.proceso.readyReadStandardError.connect(lambda: self._leer(self.proceso, True))
        self.proceso.finished.connect(lambda code, _s: self._ejecucion_terminada(code, p))
        self.proceso.errorOccurred.connect(self._proceso_error)
        self.proceso.start(programa, argumentos)
        if entrada:
            self.proceso.write(entrada.encode("utf-8"))
        self.proceso.closeWriteChannel()

    def _ejecucion_terminada(self, code: int, p: Programa):
        self.proceso = None
        self._ocupado(False)
        self._log(f"■ Terminó con código {code}.\n", C_OK if code == 0 else C_ERROR)
        self.statusBar().showMessage(f"{p.titulo}: terminó (código {code}).")
        if self.carpeta_resultados:
            self._cargar_resultados(self.carpeta_resultados)

    def detener(self):
        if self.proceso is not None:
            self.proceso.kill()
            self._log("■ Detenido por el usuario.\n", C_AVISO)

    def _proceso_error(self, error):
        # Solo FailedToStart se reporta aquí; los demás casos los cubre 'finished'.
        if error == QProcess.FailedToStart:
            self._log("✘ No se pudo iniciar el proceso (¿existe el ejecutable?).\n", C_ERROR)
            self.proceso = None
            self.compilando = False
            self._ocupado(False)

    def _leer(self, proc: QProcess, es_error: bool):
        """Vuelca a la consola lo que el proceso escribió (error en ámbar)."""
        if proc is None:
            return
        canal = proc.readAllStandardError() if es_error else proc.readAllStandardOutput()
        datos = bytes(canal).decode("utf-8", errors="replace")
        if datos:
            self._log(datos, C_AVISO if es_error else None)

    def _ocupado(self, si: bool):
        self.b_ejecutar.setEnabled(not si)
        self.b_compilar.setEnabled(not si)
        self.b_detener.setEnabled(si)

    # ---------------- consola ----------------
    def _log(self, texto: str, color: str | None = None):
        cur = self.consola.textCursor()
        cur.movePosition(QTextCursor.End)
        f = QTextCharFormat()
        f.setForeground(QColor(color or C_TEXTO))
        cur.insertText(texto, f)
        self.consola.setTextCursor(cur)
        self.consola.ensureCursorVisible()

    def _error(self, titulo: str, mensaje: str):
        QMessageBox.warning(self, titulo, mensaje)
        self._log(f"✘ {titulo}: {mensaje}\n", C_ERROR)

    # ---------------- resultados ----------------
    def _limpiar_resultados(self):
        self.lista_res.clear()
        self.pixmap_res = None
        self.img_res.setPixmap(QPixmap())
        self.img_res.setText("Aún no hay resultados.\nEjecuta el programa para ver lo que genera.")
        self.pila_res.setCurrentWidget(self.img_res)

    def _cargar_resultados(self, carpeta: Path):
        """Lista los archivos creados o modificados durante la última ejecución."""
        self._limpiar_resultados()
        nuevos = [f for f in sorted(carpeta.iterdir())
                  if f.is_file() and f.stat().st_mtime >= self.t_inicio - 1
                  and f.suffix.lower() in EXT_IMAGEN | EXT_TEXTO]
        for f in nuevos:
            self.lista_res.addItem(f.name)
            self.lista_res.item(self.lista_res.count() - 1).setData(Qt.UserRole, str(f))
        if nuevos:
            self.lista_res.setCurrentRow(0)
            self.pestanas.setCurrentIndex(1)
            self._log(f"Archivos generados: {', '.join(f.name for f in nuevos)}\n", C_OK)

    def _mostrar_resultado(self, fila: int):
        item = self.lista_res.item(fila)
        if item is None:
            return
        ruta = Path(item.data(Qt.UserRole))
        if ruta.suffix.lower() in EXT_IMAGEN:
            self.pixmap_res = QPixmap(str(ruta))
            self.pila_res.setCurrentWidget(self.img_res)
            self._escalar_resultado()
        else:
            self.pixmap_res = None
            self.txt_res.setPlainText(ruta.read_text(encoding="utf-8", errors="replace"))
            self.pila_res.setCurrentWidget(self.txt_res)

    def _escalar_resultado(self):
        if self.pixmap_res and not self.pixmap_res.isNull():
            self.img_res.setPixmap(self.pixmap_res.scaled(
                self.img_res.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._escalar_resultado()

    def _abrir_salidas(self):
        if self.actual is None:
            return
        carpeta = SALIDAS / self.actual.id
        carpeta.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(carpeta)))

    def closeEvent(self, e):
        if self.proceso is not None:
            self.proceso.kill()
        super().closeEvent(e)


# ==========================================================================
# Tema oscuro
# ==========================================================================
def aplicar_tema(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(C_FONDO))
    pal.setColor(QPalette.WindowText, QColor(C_TEXTO))
    pal.setColor(QPalette.Base, QColor("#17181c"))
    pal.setColor(QPalette.AlternateBase, QColor(C_PANEL))
    pal.setColor(QPalette.Text, QColor(C_TEXTO))
    pal.setColor(QPalette.Button, QColor(C_PANEL))
    pal.setColor(QPalette.ButtonText, QColor(C_TEXTO))
    pal.setColor(QPalette.Highlight, QColor(C_ACENTO))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.PlaceholderText, QColor(C_APAGADO))
    app.setPalette(pal)
    app.setStyleSheet(f"""
        QPushButton {{ padding:6px 14px; border-radius:5px; border:1px solid #3a3d46; }}
        QPushButton:hover {{ border-color:{C_ACENTO}; }}
        QPushButton:disabled {{ color:{C_APAGADO}; }}
        QPushButton#primario {{ background:{C_ACENTO}; color:white; border:none; font-weight:600; }}
        QPushButton#primario:disabled {{ background:#3a4a6a; color:#9aa6bd; }}
        QGroupBox {{ border:1px solid #3a3d46; border-radius:6px; margin-top:10px; padding:10px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left:10px; padding:0 4px; }}
        QTreeWidget, QListWidget, QPlainTextEdit, QLineEdit {{ border:1px solid #33363e; border-radius:4px; }}
    """)


def main():
    app = QApplication(sys.argv)
    aplicar_tema(app)
    v = Ventana()
    v.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
