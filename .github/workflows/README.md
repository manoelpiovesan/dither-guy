# GitHub Actions Workflow - Build Dither Guy Executables

Este workflow automatiza a criação de executáveis para os aplicativos Dither Guy e Dither Guy Video em três plataformas: Windows, Linux e macOS.

## Como funciona

O workflow é acionado quando você cria uma nova tag de versão (começando com 'v') ou manualmente através do GitHub Actions.

### Quando é executado?

1. **Automaticamente**: Quando você cria e envia uma tag de versão (ex: `v1.0.0`, `v2.1.3`)
2. **Manualmente**: Através da aba "Actions" no GitHub, usando o botão "Run workflow"

### Como criar uma nova release

Para criar uma nova release e gerar os executáveis automaticamente:

```bash
# 1. Commit suas mudanças
git add .
git commit -m "Release version 1.0.0"

# 2. Crie uma tag de versão
git tag v1.0.0

# 3. Envie a tag para o GitHub
git push origin v1.0.0
```

Ou crie a tag diretamente no GitHub através da interface web em "Releases".

### O que o workflow faz?

1. **Build para Windows**: Cria `DitherGuy.exe` e `DitherGuyVideo.exe`
2. **Build para Linux**: Cria `DitherGuy` e `DitherGuyVideo`
3. **Build para macOS**: Cria `DitherGuy` e `DitherGuyVideo`
4. **Cria Release**: Quando executado por uma tag, cria automaticamente uma release no GitHub com:
   - Todos os 6 executáveis anexados como assets
   - Notas de release geradas automaticamente
   - Versão não-draft e não-prerelease

### Executáveis gerados

#### Windows
- `DitherGuy.exe` - Aplicativo principal de dithering
- `DitherGuyVideo.exe` - Aplicativo de dithering para vídeos

#### Linux
- `DitherGuy` - Aplicativo principal de dithering
- `DitherGuyVideo` - Aplicativo de dithering para vídeos

#### macOS
- `DitherGuy` - Aplicativo principal de dithering
- `DitherGuyVideo` - Aplicativo de dithering para vídeos

### Verificar o progresso

Você pode acompanhar o progresso do workflow em:
`https://github.com/[seu-usuario]/dither-guy/actions`

### Troubleshooting

Se o workflow falhar:
1. Verifique os logs na aba "Actions"
2. Certifique-se de que todas as dependências estão listadas em `requirements.txt`
3. Verifique se os arquivos `dither_guy.py` e `dither_guy_video.py` existem
4. Confirme que o `app_icon.png` está presente no repositório
