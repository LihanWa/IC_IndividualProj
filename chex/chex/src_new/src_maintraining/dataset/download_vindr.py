import os
import subprocess

def download_vindr_cxr_dataset(target_dir, username=None, password=None):
    """
    下载VinDr-CXR数据集到指定目录
    
    参数:
    - target_dir: 保存数据集的目标目录
    - username: Physionet用户名
    - password: Physionet密码
    """
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 数据集ID
    dataset_id = "vindr-cxr/1.0.0"
    
    # 安装physionet-client（如果尚未安装）
    try:
        subprocess.run(["pip", "install", "physionet-client"], check=True)
    except subprocess.CalledProcessError:
        print("无法安装physionet-client，请手动安装")
        return False
    
    # 构建下载命令
    cmd = ["physionet-client", "get", dataset_id, "-r", "-d", target_dir]
    
    # 如果提供了用户名和密码，添加到命令中
    if username and password:
        cmd.extend(["-u", username, "-p", password])
    
    # 执行下载
    try:
        print(f"开始下载VinDr-CXR数据集到 {target_dir}")
        subprocess.run(cmd, check=True)
        print("下载完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"下载失败: {e}")
        print("请尝试手动下载或检查凭据")
        return False

if __name__ == "__main__":
    # 设置保存数据集的目录
    VINDR_CXR_DIR = "path/to/your/data/directory"
    
    # 你的Physionet凭据
    username = "lihanr"
    password = "@Zr1202"
    
    download_vindr_cxr_dataset(VINDR_CXR_DIR, username, password)