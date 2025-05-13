import io
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS # Importe o CORS
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)
CORS(app) # Habilita o CORS para todas as rotas e origens por padrão.
          # Para produção, você pode querer ser mais específico:
          # CORS(app, resources={r"/reconhecer": {"origins": "http://127.0.0.1:5500"}})


IMG_WIDTH = 160  # Aumentado um pouco para melhor visualização
IMG_HEIGHT = 260 # Aumentado um pouco para melhor visualização
PADDING = 20
LINE_THICKNESS = 3
BG_COLOR = (255, 255, 255)  
FG_COLOR = (0, 0, 0)        

STEM_X = IMG_WIDTH // 2
STEM_Y_START = PADDING
STEM_Y_END = IMG_HEIGHT - PADDING
STEM_Y_MID = (STEM_Y_START + STEM_Y_END) // 2

STROKE_LENGTH = min(IMG_WIDTH // 3, IMG_HEIGHT // 6)


def identificar_quadrante(x1, y1, x2, y2, cx, cy, margem):
    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2
    if ym < cy - margem: # Metade Superior
        if xm < cx - margem: # Esquerda
            return "dezena"  # Superior Esquerdo
        elif xm > cx + margem: # Direita
            return "unidade" # Superior Direito
    elif ym > cy + margem: # Metade Inferior
        if xm < cx - margem: # Esquerda
            return "milhar"  # Inferior Esquerdo
        elif xm > cx + margem: # Direita
            return "centena" # Inferior Direito
    return None

def identificar_valor_por_quadrante(quadrantes):
    valores = {"milhar": 0, "centena": 0, "dezena": 0, "unidade": 0}
    # Ordem correta para numerais cistercienses (baseado em imagens comuns do sistema)
    # Unidade (top-right), Dezena (top-left), Centena (bottom-right), Milhar (bottom-left)
    mapeamento_potencia = {"unidade": 0, "dezena": 1, "centena": 2, "milhar": 3}

    for quad, linhas in quadrantes.items():
        qtd = len(linhas)
        if 1 <= qtd <= 9:
            if quad in mapeamento_potencia:
                valores[quad] = qtd * (10 ** mapeamento_potencia[quad])
    return sum(valores.values())

def detectar_cisterciense(imagem_path):
    img = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erro ao carregar a imagem: {imagem_path}")
        return 0

    # Aplicar um desfoque gaussiano pode ajudar a suavizar a imagem e melhorar a detecção de bordas/linhas
    # blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    # _, bin_img = cv2.threshold(blurred_img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, bin_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV) # Mantenha simples por enquanto

    # Parâmetros do Canny e HoughLinesP são cruciais e podem precisar de ajuste fino
    edges = cv2.Canny(bin_img, 50, 150, apertureSize=3)
    # Tente ajustar threshold, minLineLength, maxLineGap
    # minLineLength: comprimento mínimo de uma linha. Não pode ser muito grande.
    # maxLineGap: lacuna máxima entre segmentos de linha para serem tratados como uma única linha.
    linhas = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=5)

    h, w = bin_img.shape
    cx, cy = w // 2, h // 2
    # A margem é para evitar que linhas exatamente no eixo sejam mal classificadas.
    # Pode ser uma porcentagem pequena do menor lado da imagem.
    margem = int(min(w, h) * 0.02) # Ajuste esta margem conforme necessário

    quadrantes = {"milhar": [], "centena": [], "dezena": [], "unidade": []}

    if linhas is not None:
        for linha in linhas:
            x1, y1, x2, y2 = linha[0]
            quad = identificar_quadrante(x1, y1, x2, y2, cx, cy, margem)
            if quad:
                quadrantes[quad].append(((x1, y1), (x2, y2)))
    else:
        print("Nenhuma linha detectada pela transformada de Hough.")

    # Para depuração, pode ser útil visualizar as linhas e quadrantes
    # img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.line(img_color, (cx, 0), (cx, h), (255,0,0), 1) # Eixo Y
    # cv2.line(img_color, (0, cy), (w, cy), (255,0,0), 1) # Eixo X
    # for quad_nome, linhas_quad in quadrantes.items():
    #     cor = (0,0,0)
    #     if quad_nome == "unidade": cor = (0,0,255) # Vermelho
    #     elif quad_nome == "dezena": cor = (0,255,0) # Verde
    #     elif quad_nome == "centena": cor = (255,0,255) # Magenta
    #     elif quad_nome == "milhar": cor = (0,255,255) # Ciano
    #     for l in linhas_quad:
    #         cv2.line(img_color, l[0], l[1], cor, 2)
    # temp_debug_path = os.path.join(os.path.dirname(imagem_path), "debug_" + os.path.basename(imagem_path))
    # cv2.imwrite(temp_debug_path, img_color)
    # print(f"Imagem de debug salva em: {temp_debug_path}")
    # print(f"Quadrantes detectados: { {k: len(v) for k, v in quadrantes.items()} }")


    return identificar_valor_por_quadrante(quadrantes)


# --- Funções para GERAR Numeral Cisterciense ---

def _desenhar_digito_na_posicao(image, digito, posicao):
    """
    Desenha um único dígito (1-9) na posição especificada.
    posicao: "unidade", "dezena", "centena", "milhar"
    """
    if not 1 <= digito <= 9:
        return # Não desenha nada para dígito 0 ou inválido

    # Coordenadas da seção baseadas na posição
    if posicao == "unidade": # Topo-Direita
        p_haste_sup = (STEM_X, STEM_Y_START)
        p_haste_inf = (STEM_X, STEM_Y_MID)
        p_ext_sup   = (STEM_X + STROKE_LENGTH, STEM_Y_START)
        p_ext_inf   = (STEM_X + STROKE_LENGTH, STEM_Y_MID)
    elif posicao == "dezena": # Topo-Esquerda
        p_haste_sup = (STEM_X, STEM_Y_START)
        p_haste_inf = (STEM_X, STEM_Y_MID)
        p_ext_sup   = (STEM_X - STROKE_LENGTH, STEM_Y_START)
        p_ext_inf   = (STEM_X - STROKE_LENGTH, STEM_Y_MID)
    elif posicao == "centena": # Baixo-Direita
        p_haste_sup = (STEM_X, STEM_Y_MID)
        p_haste_inf = (STEM_X, STEM_Y_END)
        p_ext_sup   = (STEM_X + STROKE_LENGTH, STEM_Y_MID)
        p_ext_inf   = (STEM_X + STROKE_LENGTH, STEM_Y_END)
    elif posicao == "milhar": # Baixo-Esquerda
        p_haste_sup = (STEM_X, STEM_Y_MID)
        p_haste_inf = (STEM_X, STEM_Y_END)
        p_ext_sup   = (STEM_X - STROKE_LENGTH, STEM_Y_MID)
        p_ext_inf   = (STEM_X - STROKE_LENGTH, STEM_Y_END)
    else:
        return

    # Desenhar formas baseadas no dígito
    if digito == 1:
        cv2.line(image, p_haste_sup, p_ext_sup, FG_COLOR, LINE_THICKNESS)
    elif digito == 2:
        cv2.line(image, p_haste_inf, p_ext_inf, FG_COLOR, LINE_THICKNESS)
    elif digito == 3:
        cv2.line(image, p_haste_sup, p_ext_inf, FG_COLOR, LINE_THICKNESS)
    elif digito == 4:
        cv2.line(image, p_haste_inf, p_ext_sup, FG_COLOR, LINE_THICKNESS)
    elif digito == 5: # 1 + 4
        _desenhar_digito_na_posicao(image, 1, posicao)
        _desenhar_digito_na_posicao(image, 4, posicao)
    elif digito == 6:
        cv2.line(image, p_ext_sup, p_ext_inf, FG_COLOR, LINE_THICKNESS)
    elif digito == 7: # 1 + 6
        _desenhar_digito_na_posicao(image, 1, posicao)
        _desenhar_digito_na_posicao(image, 6, posicao)
    elif digito == 8: # 2 + 6
        _desenhar_digito_na_posicao(image, 2, posicao)
        _desenhar_digito_na_posicao(image, 6, posicao)
    elif digito == 9: # 1 + 2 + 6 (ou 1 + 8)
        _desenhar_digito_na_posicao(image, 1, posicao)
        _desenhar_digito_na_posicao(image, 8, posicao) # Mais simples que chamar 1, 2 e 6

def gerar_imagem_cisterciense(numero):
    """
    Gera uma imagem NumPy do numeral cisterciense para o número dado.
    """
    if not 0 <= numero <= 9999: # Permite 0 como imagem vazia (apenas haste)
        raise ValueError("O número deve estar entre 0 e 9999.")

    # Cria imagem base
    image = np.full((IMG_HEIGHT, IMG_WIDTH, 3), BG_COLOR, dtype=np.uint8)

    # Desenha a haste vertical central
    cv2.line(image, (STEM_X, STEM_Y_START), (STEM_X, STEM_Y_END), FG_COLOR, LINE_THICKNESS)

    if numero == 0: # Se for 0, retorna apenas a haste
        return image

    # Decompor o número
    unidade = numero % 10
    dezena = (numero // 10) % 10
    centena = (numero // 100) % 10
    milhar = (numero // 1000) % 10

    # Desenhar cada componente
    if unidade > 0:
        _desenhar_digito_na_posicao(image, unidade, "unidade")
    if dezena > 0:
        _desenhar_digito_na_posicao(image, dezena, "dezena")
    if centena > 0:
        _desenhar_digito_na_posicao(image, centena, "centena")
    if milhar > 0:
        _desenhar_digito_na_posicao(image, milhar, "milhar")

    return image



@app.route("/gerar_cisterciense/<int:numero_decimal>", methods=["GET"])
def gerar_cisterciense_api(numero_decimal):
    try:
        imagem_np = gerar_imagem_cisterciense(numero_decimal)
        
        # Codificar imagem para PNG e enviar
        is_success, buffer = cv2.imencode(".png", imagem_np)
        if not is_success:
            return jsonify({"erro": "Falha ao codificar a imagem"}), 500
        
        img_io = io.BytesIO(buffer)
        img_io.seek(0) # Vá para o início do stream
        
        return send_file(
            img_io,
            mimetype='image/png',
            as_attachment=False, # Para exibir no navegador, não baixar
            download_name=f'cisterciense_{numero_decimal}.png' # Nome sugerido se o usuário salvar
        )
    except ValueError as ve:
        return jsonify({"erro": str(ve)}), 400
    except Exception as e:
        print(f"Exceção durante a geração da imagem: {e}")
        return jsonify({"erro": "Erro interno ao gerar a imagem"}), 500


@app.route("/reconhecer", methods=["POST"])
def reconhecer():
    if 'imagem' not in request.files:
        return jsonify({"erro": "Nenhuma imagem enviada"}), 400

    imagem = request.files['imagem']
    if imagem.filename == '':
        return jsonify({"erro": "Nome de arquivo inválido"}), 400

    temp_file_path = ""
    try:
        # Salvar arquivo temporário de forma segura
        _, ext = os.path.splitext(imagem.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".png") as tmp:
            imagem.save(tmp.name)
            temp_file_path = tmp.name
        
        # É crucial fechar o arquivo antes que outra função (como cv2.imread) tente abri-lo,
        # especialmente no Windows. Como usamos 'with', ele é fechado ao sair do bloco.
        # No entanto, NamedTemporaryFile no Windows não pode ser reaberto por nome
        # se delete=True (padrão). Com delete=False, precisamos garantir que o arquivo seja excluído.

        resultado = detectar_cisterciense(temp_file_path)
        return jsonify({"numero": resultado})

    except Exception as e:
        print(f"Exceção durante o reconhecimento: {e}") # Log do erro no servidor
        # import traceback
        # traceback.print_exc() # Para um stacktrace completo no log do servidor
        return jsonify({"erro": f"Erro interno no processamento da imagem: {str(e)}"}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


if __name__ == "__main__":
    # Você pode especificar o host e a porta aqui se necessário
    app.run(debug=True, host='0.0.0.0', port=5000) # '0.0.0.0' torna acessível na rede local