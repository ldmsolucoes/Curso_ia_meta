import os
import zipfile
import shutil
import pandas as pd
import unicodedata

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def menu_principal():
    while True:
        print("\nMenu Principal:")
        print("1- Receber arquivos de NF-e (.zip)")
        print("2- Criar/Recriar base de conhecimentos NF-e")
        print("3- Pesquise sobre suas NF-E")
        print("0- Sair")
        escolha = input("Escolha uma opção: ")
        if escolha == "1":
            sucesso, msg = receber_arquivos_nfe()
            print(msg)
        elif escolha == "2":
            criar_base_conhecimento()
        elif escolha == "3":
            pesquisar_nfe()
        elif escolha == "0":
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")

def receber_arquivos_nfe():
    zip_path = input("Informe o caminho do arquivo .zip de NF-e: ")
    output_dir = "NFs_Extraidas"
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        return True, "Arquivo descompactado com sucesso!"
    except Exception as e:
        return False, f"Erro ao descompactar: {str(e)}"

def padronizar_nome_coluna(col):
    col = ''.join((c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn'))
    col = col.strip().upper()
    col = col.replace('/', '_').replace(' ', '_')
    return col

def converter_para_utf8(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='latin1') as f:
        conteudo = f.read()
    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        f.write(conteudo)

def criar_base_conhecimento():
    import os
    import shutil
    import pandas as pd
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    # Caminhos dos arquivos CSV
    pasta_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NFs_Extraidas")
    cabecalho_path = os.path.join(pasta_csv, "202401_NFs_Cabecalho.csv")
    itens_path = os.path.join(pasta_csv, "202401_NFs_Itens.csv")

    # Remove o banco anterior, se existir
    if os.path.exists("./nfe_db"):
        shutil.rmtree("./nfe_db")

    # Converte os arquivos para UTF-8 antes de ler
    converter_para_utf8(cabecalho_path)
    converter_para_utf8(itens_path)

    # Lê os CSVs já em UTF-8
    df_cab = pd.read_csv(cabecalho_path, encoding='utf-8')
    df_itens = pd.read_csv(itens_path, encoding='utf-8')

    # Padroniza os nomes das colunas
    df_cab.columns = [padronizar_nome_coluna(col) for col in df_cab.columns]
    df_itens.columns = [padronizar_nome_coluna(col) for col in df_itens.columns]

    documentos = []

    for _, row in df_cab.iterrows():
        chave = row['CHAVE_DE_ACESSO']
        itens_nota = df_itens[df_itens['CHAVE_DE_ACESSO'] == chave]

        itens_texto = ""
        for _, item in itens_nota.iterrows():
            itens_texto += (
                f"\n  - Produto: {item.get('DESCRICAO_DO_PRODUTO_SERVICO', '')}, "
                f"Qtd: {item.get('QUANTIDADE', '')}, "
                f"Valor Total: {item.get('VALOR_TOTAL', '')}"
            )

        doc = (
            f"Nota Fiscal: {row.get('NUMERO', '')}\n"
            f"Chave de Acesso: {chave}\n"
            f"Emitente: {row.get('RAZAO_SOCIAL_EMITENTE', '')}\n"
            f"Valor Total: {row.get('VALOR_NOTA_FISCAL', '')}\n"
            f"Itens:{itens_texto}\n"
        )
        documentos.append(doc)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_texts(documentos, embeddings, persist_directory="./nfe_db")
    # Não chame vector_store.persist()! O Chroma já salva automaticamente.

    print(f"Base de conhecimento criada com {len(documentos)} notas fiscais.")

# --- SEPARAÇÃO: funções para ler notas e itens, preservando lógica original ---

def carregar_notas():
    colunas_cabecalho = [
        'CHAVE_DE_ACESSO', 'MODELO', 'SERIE', 'NUMERO', 'NATUREZA_DA_OPERACAO',
        'DATA_EMISSAO', 'EVENTO_MAIS_RECENTE', 'DATA_HORA_EVENTO_MAIS_RECENTE',
        'CPF_CNPJ_EMITENTE', 'RAZAO_SOCIAL_EMITENTE', 'INSCRICAO_ESTADUAL_EMITENTE',
        'UF_EMITENTE', 'MUNICIPIO_EMITENTE', 'CNPJ_DESTINATARIO', 'NOME_DESTINATARIO',
        'UF_DESTINATARIO', 'INDICADOR_IE_DESTINATARIO', 'DESTINO_DA_OPERACAO',
        'CONSUMIDOR_FINAL', 'PRESENCA_DO_COMPRADOR', 'VALOR_NOTA_FISCAL'
    ]
    df_cab = pd.read_csv(
        "NFs_Extraidas/202401_NFs_Cabecalho.csv",
        encoding='utf-8',
        dtype=str,
        names=colunas_cabecalho,
        header=0
    )
    df_cab.columns = [padronizar_nome_coluna(col) for col in df_cab.columns]
    return df_cab

def carregar_itens():
    # Lê o CSV de itens usando o cabeçalho real do arquivo, sem forçar nomes
    df_itens = pd.read_csv(
        "NFs_Extraidas/202401_NFs_Itens.csv",
        encoding='utf-8',
        dtype=str,
        header=0  # Usa o cabeçalho real do arquivo
     )    
    # Padroniza os nomes das colunas para manter compatibilidade
    df_itens.columns = [padronizar_nome_coluna(col) for col in df_itens.columns]
    return df_itens

def filtrar_por_substring_em_todas_colunas(df, termo):
    termo = termo.lower()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        mask[col] = df[col].astype(str).str.lower().str.contains(termo, na=False)
    linhas_filtradas = mask.any(axis=1)
    return df.loc[linhas_filtradas]

def pesquisar_nfe():
    print("\nPesquisa de NF-e")
    print("Digite sua pergunta ou termo de busca conforme abaixo:")
    print("- Para buscar uma nota fiscal, digite: nota <numero>")
    print("  Exemplo: nota 369180")
    print("- Para buscar uma nota pelo número da chave de acesso, digite: chave de acesso <chave>")
    print("  Exemplo: chave de acesso 50240129843878000170550010000025251000181553")
    print("- Para buscar os itens de uma nota, digite: itens <numero ou chave ou termo>")
    print("  Exemplo: itens 2525")
    print("           itens 50240129843878000170550010000025251000181553")
    print("           itens descrição_parcial_do_produto(ainda não implementado")
    print("- Para buscar por emitente, digite: emitente <nome>")
    print("  Exemplo: emitente supermercado xyz")
    print(" - A busca Semantica está muito genérica, falta refinar...")
    consulta = input("Faça sua pergunta: ").strip().lower()

    try:
        df_cab = carregar_notas()
        df_itens = carregar_itens()
    except Exception as e:
        print(f"Erro ao carregar dados tabulares: {e}")
        return

    if consulta.startswith("nota "):
        numero = consulta.split(" ", 1)[1].strip()
        resultados_exatos = df_cab[df_cab['NUMERO'].str.contains(numero, na=False)].to_dict('records')
        if resultados_exatos:
            for i, nota in enumerate(resultados_exatos, 1):
                print(f"\nResultado {i}:")
                print(f"Nota Fiscal: {nota.get('NUMERO', '(valor vazio)')}")
                print(f"Chave de Acesso: {nota.get('CHAVE_DE_ACESSO', '(valor vazio)')}")
                print(f"Emitente: {nota.get('RAZAO_SOCIAL_EMITENTE', '(valor vazio)')}")
                print(f"Valor Total: {nota.get('VALOR_NOTA_FISCAL', '(valor vazio)')}")
        else:
            print("Nenhuma nota encontrada com esse número.")

    elif consulta.startswith("chave de acesso "):
        chave = consulta.split(" ", 3)[3].strip()
        resultados_exatos = df_cab[df_cab['CHAVE_DE_ACESSO'].str.contains(chave, na=False)].to_dict('records')
        if resultados_exatos:
            for i, nota in enumerate(resultados_exatos, 1):
                print(f"\nResultado {i}:")
                print(f"Nota Fiscal: {nota.get('NUMERO', '(valor vazio)')}")
                print(f"Chave de Acesso: {nota.get('CHAVE_DE_ACESSO', '(valor vazio)')}")
                print(f"Emitente: {nota.get('RAZAO_SOCIAL_EMITENTE', '(valor vazio)')}")
                print(f"Valor Total: {nota.get('VALOR_NOTA_FISCAL', '(valor vazio)')}")
        else:
            print("Nenhuma nota encontrada com essa chave de acesso.")

    elif consulta.startswith("emitente "):
        termo = consulta.split(" ", 1)[1].strip()
        resultados_emitente = df_cab[df_cab['RAZAO_SOCIAL_EMITENTE'].str.lower().str.contains(termo, na=False)].to_dict('records')
        if resultados_emitente:
            for i, nota in enumerate(resultados_emitente, 1):
                print(f"\nResultado {i}:")
                print(f"Nota Fiscal: {nota.get('NUMERO', '(valor vazio)')}")
                print(f"Chave de Acesso: {nota.get('CHAVE_DE_ACESSO', '(valor vazio)')}")
                print(f"Emitente: {nota.get('RAZAO_SOCIAL_EMITENTE', '(valor vazio)')}")
                print(f"Valor Total: {nota.get('VALOR_NOTA_FISCAL', '(valor vazio)')}")
        else:
            print("Nenhuma nota encontrada para esse emitente.")

    elif consulta.startswith("itens "):
        termo = consulta.split(" ", 1)[1].strip()
        termo_limpo = termo.replace(" ", "")
        df_itens['CHAVE_DE_ACESSO'] = df_itens['CHAVE_DE_ACESSO'].astype(str).str.replace(" ", "").str.strip()

        # Detecta dinamicamente a coluna de descrição do produto
        col_produto = None
        for col in df_itens.columns:
            if 'PRODUTO' in col and ('DESCR' in col or 'DESCRICAO' in col):
                col_produto = col
                break

        chaves = []
        # Busca por chave de acesso (44 dígitos)
        if termo_limpo.isdigit() and len(termo_limpo) == 44:
            chaves = [termo_limpo]
        # Busca por número da nota (ex: itens 2525)
        elif termo_limpo.isdigit():
            chaves = df_cab[df_cab['NUMERO'].str.contains(termo_limpo, na=False)]['CHAVE_DE_ACESSO'].tolist()

        if chaves:
            for chave in chaves:
                nota_info = df_cab[df_cab['CHAVE_DE_ACESSO'] == chave]
                if not nota_info.empty:
                    nota = nota_info.iloc[0]
                    print(f"\nNota Fiscal: {nota.get('NUMERO', '(valor vazio)')}")
                    print(f"Chave de Acesso: {nota.get('CHAVE_DE_ACESSO', '(valor vazio)')}")
                    print(f"Emitente: {nota.get('RAZAO_SOCIAL_EMITENTE', '(valor vazio)')}")
                    print(f"Valor Total da Nota: {nota.get('VALOR_NOTA_FISCAL', '(valor vazio)')}")
                itens = df_itens[df_itens['CHAVE_DE_ACESSO'] == chave]
                if itens.empty:
                    print("Nenhum item encontrado para essa nota.")
                else:
                    valor_total_itens = 0.0
                    print("Itens:")
                    for _, item in itens.iterrows():
                        produto = item[col_produto] if col_produto and col_produto in item else ''
                        quantidade = item['QUANTIDADE'] if 'QUANTIDADE' in item else ''
                        valor_total = item['VALOR_TOTAL'] if 'VALOR_TOTAL' in item else ''
                        try:
                            valor_total_itens += float(str(valor_total).replace(',', '.'))
                        except Exception:
                            pass
                        print(f"- Produto: {produto}, Quantidade: {quantidade}, Valor Total: {valor_total}")
                    print(f"Soma dos valores dos itens: {valor_total_itens}")
        else:
            # Busca genérica por substring em todas as colunas dos itens
            itens = filtrar_por_substring_em_todas_colunas(df_itens, termo)
            if itens.empty:
                print("Nenhum item encontrado para essa consulta.")
            else:
                print(f"\nItens encontrados para a consulta '{consulta}':")
                for _, item in itens.iterrows():
                    produto = item[col_produto] if col_produto and col_produto in item else ''
                    quantidade = item['QUANTIDADE'] if 'QUANTIDADE' in item else ''
                    valor_total = item['VALOR_TOTAL'] if 'VALOR_TOTAL' in item else ''
                    print(f"- Produto: {produto}, Quantidade: {quantidade}, Valor Total: {valor_total}")



    else:
        # Fallback: busca semântica usando embeddings e Chroma
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = Chroma(persist_directory="./nfe_db", embedding_function=embeddings)
            docs = vector_store.similarity_search(consulta, k=3)
            if docs:
                print("\nResultados semânticos encontrados:")
                for i, doc in enumerate(docs, 1):
                    print(f"\nResultado {i}:\n{doc.page_content}")
            else:
                print("Nenhum resultado encontrado para sua consulta.")
        except Exception as e:
            print("Comando não reconhecido. Por favor, use os formatos indicados no prompt.")
            print(f"(Erro semântico: {e})")
if __name__ == "__main__":
    menu_principal()
