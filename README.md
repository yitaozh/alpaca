## Manual

### How to setup the env

#### On new computer

1. Get the .env from somewhere else
2. Install pyenv
    Add `eval "$(pyenv init -)"` to `.zshrc` or `.bashrc`
3. Install Python 3.9
    ```
    pyenv install 3.9.18
    pyenv local 3.9.18   # 在项目目录下固定版本
    python --version     # 应该显示 3.9.18
    ```
4. Activate env
    ```
    python -m venv venv && source venv/bin/activate
    ```
5. Install libs
    ```
    pip install -r requirements.txt
    ```

#### Already setup

```
source venv/bin/activate
```