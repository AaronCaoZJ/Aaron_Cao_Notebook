# 解决离线安装 VSCode Extension 时版本不匹配问题

## 下载 `.vsix`格式的安装包

VSCode 插件安装包下载链接：

`https://marketplace.visualstudio.com/`

## 使用压缩软件打开安装包

>  *注意不要解压安装包后再打开文件，要在压缩包文件管理软件中直接打开修改*

![image-20240317211451813](assets/image-20240317211451813-1710681300953-1.png)

![image-20240317211709280](assets/image-20240317211709280-1710681438437-3.png)

## 修改配置文件中对 VSCode 版本的设置

在打开的`package.json`中搜索`engines`

![image-20240317211839585](assets/image-20240317211839585-1710681522584-5.png)

此处使用`Ubuntu`上的 VSCode，版本`1.87.2`，如上图修改并应用修改，保存

## 重新使用`Install from VSIX`离线安装插件即可

![image-20240317212727014](assets/image-20240317212727014-1710682050120-7.png)

