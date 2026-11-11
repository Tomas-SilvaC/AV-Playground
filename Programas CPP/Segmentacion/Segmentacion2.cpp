/**
 * Segmentacion2.cpp
 * Segmentación con ideas de CNN y Vision Transformer (ViT), sin pesos entrenados.
 *
 * ## Idea general
 * Este programa reproduce, de forma didáctica y sin entrenamiento, las etapas
 * fundamentales de dos arquitecturas de visión:
 *
 * ### CNN (red neuronal convolucional)
 *   convolución -> ReLU -> max-pooling
 *   - La convolución aplica filtros (aquí kernels clásicos: Sobel X/Y, Laplaciano y
 *     Gauss) para extraer características locales (bordes, suavizado).
 *   - ReLU descarta las respuestas negativas (no linealidad).
 *   - El max-pooling reduce la resolución a la mitad quedándose con el máximo de
 *     cada bloque 2x2, conservando los rasgos más destacados.
 *   En una CNN real los kernels se aprenden; aquí se fijan a mano.
 *
 * ### ViT (Vision Transformer)
 *   1) Fragmentación: la imagen se divide en parches PxP sin solaparse y cada uno
 *      se aplana a un vector (tokens, análogos a las palabras en NLP).
 *   2) Embedding lineal: cada vector se proyecta a dimensión D. Aquí la matriz de
 *      proyección es aleatoria y fija (en un ViT real es aprendida).
 *   3) Codificación posicional: se suma información de posición (sinusoidal), porque
 *      la atención por sí sola no conoce el orden espacial de los parches.
 *   4) Token CLS: token extra que resume la imagen completa (aquí, la media de los
 *      parches; en un ViT real es un vector aprendible).
 *   5) Autoatención de una cabeza: softmax(Q K^T / T) V con Q = K = V. Cada token se
 *      relaciona con todos los demás, capturando contexto global desde la primera
 *      capa (a diferencia de la convolución, que es local). T actúa de temperatura.
 *
 * ### Segmentación
 *   Con los tokens ya contextualizados se agrupan con k-means (k grupos): parches
 *   parecidos reciben la misma etiqueta, y la etiqueta se pinta como máscara sobre
 *   la imagen. Es una segmentación a nivel de parche (no de píxel).
 *
 * ## Uso
 *   vit_seg imagen.jpg [parche=16] [k=4] [--show]
 *   o sin argumentos: el programa pide la ruta por consola
 *
 * ## Salidas
 *   out_panel_vit.png, out_segmentacion.png, out_atencion.png, out_cnn_features.png
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>

using namespace cv;
using namespace std;

static const int   IMG_SIZE = 224;   // tamano de entrada del ViT clasico
static const int   EMB_DIM  = 64;    // dimension D del embedding
static const float TEMP     = 0.05f; // temperatura del softmax (sobre coseno)
static const float PE_SCALE = 0.3f;  // peso de la codificacion posicional

// ---------- CNN: convolucion + ReLU + max pooling ----------
// Max-pooling 2x2 con paso 2: cada bloque de 2x2 píxeles se reemplaza por su valor
// máximo, reduciendo ancho y alto a la mitad. Entrada y salida son CV_32F.

static Mat maxPool2x2(const Mat& src) {
    Mat dst(src.rows / 2, src.cols / 2, CV_32F);
    for (int y = 0; y < dst.rows; ++y)
        for (int x = 0; x < dst.cols; ++x) {
            float m = src.at<float>(2*y, 2*x);
            m = max(m, src.at<float>(2*y,   2*x+1));
            m = max(m, src.at<float>(2*y+1, 2*x));
            m = max(m, src.at<float>(2*y+1, 2*x+1));
            dst.at<float>(y, x) = m;
        }
    return dst;
}

// Aplica una "capa CNN" fija con 4 filtros (Sobel X, Sobel Y, Laplaciano, Gauss 5x5)
// y devuelve un mosaico 2x2 con los mapas de características resultantes.
// Por cada filtro: convolución (filter2D) -> ReLU -> max-pooling -> normalizar para
// visualizar -> reescalar a IMG_SIZE x IMG_SIZE.
// @param bgr  Imagen de entrada (ya redimensionada a IMG_SIZE).

static Mat cnnFeatureMaps(const Mat& bgr) {
    Mat gray, g32;
    cvtColor(bgr, gray, COLOR_BGR2GRAY);
    gray.convertTo(g32, CV_32F, 1.0 / 255.0);

    static const float sx[9] = {-1,0,1, -2,0,2, -1,0,1};
    static const float sy[9] = {-1,-2,-1, 0,0,0, 1,2,1};
    static const float lp[9] = {0,1,0, 1,-4,1, 0,1,0};
    Mat gk = getGaussianKernel(5, 1.2, CV_32F);
    vector<pair<string, Mat>> kernels = {
        {"Sobel X",    Mat(3, 3, CV_32F, (void*)sx).clone()},
        {"Sobel Y",    Mat(3, 3, CV_32F, (void*)sy).clone()},
        {"Laplaciano", Mat(3, 3, CV_32F, (void*)lp).clone()},
        {"Gauss 5x5",  Mat(gk * gk.t())},
    };

    vector<Mat> tiles;
    for (auto& [name, k] : kernels) {
        Mat conv, relu;
        filter2D(g32, conv, CV_32F, k, Point(-1,-1), 0, BORDER_REFLECT);  // convolucion
        relu = max(conv, 0.0f);                                           // ReLU
        Mat pooled = maxPool2x2(relu);                                    // pooling
        normalize(pooled, pooled, 0, 255, NORM_MINMAX);
        Mat u8; pooled.convertTo(u8, CV_8U);
        resize(u8, u8, Size(IMG_SIZE, IMG_SIZE), 0, 0, INTER_NEAREST);
        cvtColor(u8, u8, COLOR_GRAY2BGR);
        putText(u8, name, Point(6, 18), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1, LINE_AA);
        tiles.push_back(u8);
    }
    Mat top, bot;
    hconcat(vector<Mat>{tiles[0], tiles[1]}, top);
    hconcat(vector<Mat>{tiles[2], tiles[3]}, bot);
    Mat out; vconcat(top, bot, out);
    return out;
}

// ---------- ViT: parches, embedding, posicion, CLS, autoatencion ----------
// Paso 1 del ViT: divide la imagen en parches PxP sin solaparse y aplana cada uno
// (P x P x 3 canales) en un vector fila de longitud P*P*3.
// @param img  Imagen cuadrada IMG_SIZE x IMG_SIZE en BGR.
// @param P    Tamaño del parche (debe dividir a IMG_SIZE).
// @return     Matriz N x (P*P*3), con N = (IMG_SIZE/P)^2 tokens, en orden fila a fila.
static Mat flattenPatches(const Mat& img, int P) {
    int g = img.rows / P, N = g * g, len = P * P * 3;
    Mat tokens(N, len, CV_32F);
    Mat f; img.convertTo(f, CV_32FC3, 1.0 / 255.0);
    for (int i = 0; i < g; ++i)
        for (int j = 0; j < g; ++j) {
            Mat patch = f(Rect(j*P, i*P, P, P)).clone();   // P x P x C
            patch.reshape(1, 1).copyTo(tokens.row(i*g + j)); // vector 1 x (P*P*C)
        }
    return tokens;
}

// Paso 3 del ViT: codificación posicional sinusoidal.
// Para la posición 'pos' y la dimensión i: seno si i es par, coseno si es impar,
// con frecuencias decrecientes según 10000^(2*(i/2)/D). Cada posición obtiene así
// un patrón único que el modelo puede usar para saber dónde estaba cada parche.
// @param N  Número de posiciones (tokens).
// @param D  Dimensión del embedding.

static Mat positionalEncoding(int N, int D) {
    Mat pe(N, D, CV_32F);
    for (int pos = 0; pos < N; ++pos)
        for (int i = 0; i < D; ++i) {
            double w = pos / pow(10000.0, 2.0 * (i / 2) / D);
            pe.at<float>(pos, i) = (i % 2 == 0) ? (float)sin(w) : (float)cos(w);
        }
    return pe;
}

// Normaliza cada fila a norma L2 = 1. Así el producto punto entre dos filas es su
// similitud coseno, que es lo que se usa como puntuación de atención.

static void l2NormalizeRows(Mat& m) {
    for (int r = 0; r < m.rows; ++r) {
        double n = norm(m.row(r));
        if (n > 1e-9) m.row(r) /= n;
    }
}

// Softmax por filas (estable numéricamente: se resta el máximo de cada fila antes
// de exponenciar). Convierte cada fila en una distribución de probabilidad.

static void softmaxRows(Mat& m) {
    for (int r = 0; r < m.rows; ++r) {
        double mx; minMaxLoc(m.row(r), nullptr, &mx);
        Mat row = m.row(r) - mx;
        exp(row, row);
        row /= sum(row)[0];
        row.copyTo(m.row(r));
    }
}

int main(int argc, char** argv) {
    string path;
    int P = 16, K = 4; bool show = false;

    if (argc < 2) {
        // Modo interactivo: pide los datos por consola
        auto ask = [](const string& msg) {
            cout << msg; string line; getline(cin, line); return line;
        };
        path = ask("Ruta de la imagen (ej. C:\\fotos\\paisaje.jpg): ");
        // quitar comillas y espacios (al arrastrar el archivo a la consola Windows las agrega)
        while (!path.empty() && (path.front() == '"' || path.front() == ' ')) path.erase(path.begin());
        while (!path.empty() && (path.back() == '"' || path.back() == ' ' || path.back() == '\r')) path.pop_back();
        string p = ask("Tamano de parche [16]: "); if (!p.empty()) P = stoi(p);
        string k = ask("Numero de grupos k [4]: "); if (!k.empty()) K = stoi(k);
        show = true;
    } else {
        path = argv[1];
        int pos = 0;
        for (int a = 2; a < argc; ++a) {
            string s = argv[a];
            if (s == "--show") show = true;
            else if (pos == 0) { P = stoi(s); ++pos; }
            else if (pos == 1) { K = stoi(s); ++pos; }
        }
    }
    // La imagen debe poder dividirse exactamente en parches PxP.
    if (IMG_SIZE % P != 0) { cerr << "El parche debe dividir " << IMG_SIZE << "\n"; return 1; }

    Mat src = imread(path);
    if (src.empty()) { cerr << "No se pudo leer: " << path << "\n"; return 1; }
    Mat img; resize(src, img, Size(IMG_SIZE, IMG_SIZE));
    int g = IMG_SIZE / P, N = g * g;
    cout << "Imagen " << IMG_SIZE << "x" << IMG_SIZE << ", parche " << P
         << "x" << P << " -> " << N << " tokens (+CLS), D=" << EMB_DIM << "\n";

    // 1) Parches aplanados  (N x P*P*3)
    Mat tokens = flattenPatches(img, P);

    // Visual de la cuadricula de parches
    Mat grid = img.clone();
    for (int i = 1; i < g; ++i) {
        line(grid, Point(i*P, 0), Point(i*P, IMG_SIZE), Scalar(0,255,0), 1);
        line(grid, Point(0, i*P), Point(IMG_SIZE, i*P), Scalar(0,255,0), 1);
    }

    // 2) Embedding lineal: proyeccion aleatoria fija W (P*P*3 x D)
    //    Semilla fija (42) para que los resultados sean reproducibles.
    RNG rng(42);
    Mat W(tokens.cols, EMB_DIM, CV_32F);
    rng.fill(W, RNG::NORMAL, 0.0, 1.0 / sqrt((double)tokens.cols));
    Mat tokCentered;
    Mat mu; reduce(tokens, mu, 0, REDUCE_AVG);
    tokCentered = tokens - repeat(mu, N, 1);       // centrar: resalta diferencias entre parches
    Mat X = tokCentered * W;                       // N x D
    l2NormalizeRows(X);

    // 3) Codificacion posicional sinusoidal (sumada, con peso PE_SCALE)
    X += PE_SCALE * positionalEncoding(N, EMB_DIM) / sqrt((float)EMB_DIM);

    // 4) Token CLS (media de los parches; en un ViT real es aprendible)
    //    Se antepone como fila 0: la secuencia pasa a tener N+1 tokens.
    Mat cls; reduce(X, cls, 0, REDUCE_AVG);
    Mat Xc; vconcat(cls, X, Xc);                   // (N+1) x D
    l2NormalizeRows(Xc);

    // 5) Autoatencion de una cabeza: softmax(Q K^T / T) V con Q=K=V=Xc
    //    scores(i, j) = cuánto "atiende" el token i al token j.
    Mat scores = (Xc * Xc.t()) / TEMP;
    softmaxRows(scores);
    Mat Y = scores * Xc;                           // tokens contextualizados

    // Mapa de atencion del CLS hacia cada parche (fila 0, columnas 1..N)
    // se muestra como mapa de calor superpuesto a la imagen.
    Mat attn(g, g, CV_32F);
    for (int i = 0; i < N; ++i) attn.at<float>(i / g, i % g) = scores.at<float>(0, i + 1);
    normalize(attn, attn, 0, 255, NORM_MINMAX);
    Mat attn8, heat;
    attn.convertTo(attn8, CV_8U);
    resize(attn8, attn8, Size(IMG_SIZE, IMG_SIZE), 0, 0, INTER_CUBIC);
    applyColorMap(attn8, heat, COLORMAP_JET);
    Mat attnOverlay; addWeighted(img, 0.5, heat, 0.5, 0, attnOverlay);

    // 6) Segmentacion: k-means sobre los tokens (sin CLS)
    //    Parches con características contextualizadas similares quedan en el mismo grupo.
    Mat feats = Y.rowRange(1, N + 1).clone();
    Mat labels, centers;
    kmeans(feats, K, labels,
           TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 1e-4),
           5, KMEANS_PP_CENTERS, centers);

    // Mapa de etiquetas (g x g) escalado a [0,255] y ampliado con vecino más cercano
    // para conservar los bloques nítidos; luego se colorea y se mezcla con la imagen.
    Mat labelMap(g, g, CV_8U);
    for (int i = 0; i < N; ++i)
        labelMap.at<uchar>(i / g, i % g) = (uchar)(labels.at<int>(i) * (255 / max(1, K - 1)));
    Mat labelBig, segColor;
    resize(labelMap, labelBig, Size(IMG_SIZE, IMG_SIZE), 0, 0, INTER_NEAREST);
    applyColorMap(labelBig, segColor, COLORMAP_TURBO);
    Mat segOverlay; addWeighted(img, 0.45, segColor, 0.55, 0, segOverlay);

    // 7) CNN: mapas de caracteristicas
    Mat cnn = cnnFeatureMaps(img);

    // Salidas
    // Escribe el texto dos veces (borde negro grueso + relleno blanco) para que
    // sea legible sobre cualquier fondo.
    auto label = [](Mat m, const string& t) {
        putText(m, t, Point(6, 18), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0,0,0), 3, LINE_AA);
        putText(m, t, Point(6, 18), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255,255,255), 1, LINE_AA);
        return m;
    };
    Mat a = label(img.clone(), "Original"), b = label(grid, "Parches " + to_string(P) + "x" + to_string(P)),
        c = label(attnOverlay, "Atencion CLS"), d = label(segOverlay, "Segmentacion k=" + to_string(K));
    Mat row1, row2, panel;
    hconcat(vector<Mat>{a, b}, row1);
    hconcat(vector<Mat>{c, d}, row2);
    vconcat(row1, row2, panel);

    imwrite("out_panel_vit.png", panel);
    imwrite("out_segmentacion.png", segOverlay);
    imwrite("out_atencion.png", attnOverlay);
    imwrite("out_cnn_features.png", cnn);
    cout << "Guardado: out_panel_vit.png, out_segmentacion.png, out_atencion.png, out_cnn_features.png\n";

    if (show) {
        imshow("ViT: original / parches / atencion / segmentacion", panel);
        imshow("CNN: conv + ReLU + pooling", cnn);
        waitKey(0);
    }
    return 0;
}
