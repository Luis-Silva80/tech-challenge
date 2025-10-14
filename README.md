## Tech-Challenge (2ª Fase)

### Otimização dos hiperparâmetros da 1ª Fase com Algoritmo Genético

### 1. Alunos (6IADT)
- Luis Gustavo de Araújo Silva — RM 366233  
- Vinicius Santos de Oliveira — RM 366276

### 2. Script principal & dataset
- **Script principal:** `main.py`  
- **Otimização dos hiperparêmtros:** `hiperparameters_otimization.py`
- **Funções do algoritmo genético:** `genetic_algorithm.py`
- **Plot de gráfico do algoritmo:** `draw_functions.py`
- **Testes automatizados do cálculo de fitness:** `test_calculate_fitness.py`
- **Dataset:** `datasets/diabetes.csv`

### 3. Bibliotecas utilizadas
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn
- pygame

### 4. Passo a passo para instalação

1. **Instale o Python (3.10 ou superior)**  
   - **Windows:** [Download Python](https://www.python.org/downloads/)  
   - **macOS/Linux:** via gerenciador de pacotes ou pelo site oficial.  

2. **Abra o terminal** na pasta do projeto (`tech-challenge`).

3. **Crie e ative um ambiente virtual:**
   - **Windows (Git Bash):**
     ```bash
     python -m venv .venv
     source .venv/Scripts/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

4. *(Opcional)* **Atualize o pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

5. **Instale as bibliotecas:**
   ```bash
   pip install -r requirements.txt
   ```

6. **Execute o script:**
   ```bash
   python main.py
   ```
   > No macOS/Linux, use `python3 main.py` se `python` não estiver associado ao Python 3.

### 5. Observações
- Certifique-se de que o dataset `datasets/diabetes.csv` existe na pasta indicada.
- No Windows/PowerShell, se houver bloqueio ao ativar o ambiente virtual, execute (como administrador):  

  ```powershell
  Set-ExecutionPolicy RemoteSigned
  ```
