# Troubleshooting ssh connection issues

## 下载与本地 VSCode 对应的 vscode server

### Win VSCode

```
Version: 1.87.0
Commit: 019f4d1419fbc8219a181fab7892ebccf7ee29a2
Date: 2024-02-27T23:41:44.469Z
Electron: 27.3.2
ElectronBuildId: 26836302
Chromium: 118.0.5993.159
Node.js: 18.17.1
V8: 11.8.172.18-electron.0
OS: Windows_NT x64 10.0.22000
```

Download from:

`https://update.code.visualstudio.com/commit:019f4d1419fbc8219a181fab7892ebccf7ee29a2/server-linux-x64/stable`

### Linux VSCode

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Version: 1.87.2
Commit: 863d2581ecda6849923a2118d93a088b0745d9d6
Date: 2024-03-08T15:14:59.643Z
Electron: 27.3.2
ElectronBuildId: 26836302
Chromium: 118.0.5993.159
Node.js: 18.17.1
V8: 11.8.172.18-electron.0
OS: Linux x64 5.15.0-100-generic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Download from:

`https://update.code.visualstudio.com/commit:863d2581ecda6849923a2118d93a088b0745d9d6/server-linux-x64/stable`



## 解压到 .vscode-server 目录

> 清空原始 vscode-server/bin
>
> ```bash
> rm ~/.vscode-server/bin/* -rf
> ```

> 创建新的 bin 目录
>
> ```bash
> mkdir -p ~/.vscode-server/bin
> cd ~/.vscode-server/bin
> ```

> 解压目标文件移动到与 Commit 同名文件夹
>
> ```bash
> tar -zxf vscode-server-linux-x64.tar.gz
> # win
> mv vscode-server-linux-x64 019f4d1419fbc8219a181fab7892ebccf7ee29a2
> # ubuntu
> mv vscode-server-linux-x64 863d2581ecda6849923a2118d93a088b0745d9d6

> 重现尝试 ssh 连接即可