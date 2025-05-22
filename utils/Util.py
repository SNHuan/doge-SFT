# Updated Util.py with improved handling for local paths and better error propagation
from pathlib import Path
from typing import Optional, Union, Any, Dict, Callable
import logging
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 定义路径常量
model_save_path = Path('./models/model')
dataset_save_path = Path('./models/dataset')
tokenizer_save_path = Path('./models/tokenizer')


def load_or_download(
    obj_type: str,
    load_func: Callable,
    save_func: Callable,
    name: str,
    path: Union[str, Path],
    local_args: Dict = None,
    remote_args: Dict = None,
) -> Optional[Union[Dataset, PreTrainedTokenizer, PreTrainedModel]]:
    """加载或下载Hugging Face资源
    
    Args:
        obj_type: 资源类型描述(如"数据集"、"分词器"等)
        load_func: 加载函数
        save_func: 保存函数
        name: 资源名称或路径
        path: 本地保存路径
        local_args: 从本地加载时传递给load_func的参数字典
        remote_args: 从远程加载时传递给load_func的参数字典
        
    Returns:
        加载的资源对象，失败时返回None
    """
    path = Path(path) if isinstance(path, str) else path
    full_path = path / name
    
    # 确保参数字典存在
    local_args = local_args or {}
    remote_args = remote_args or {}
    
    try:
        if full_path.exists():
            logger = logging.getLogger(f"load.{obj_type}")
            logger.info(f"从本地加载: {full_path}")
            # 传递本地参数
            obj = load_func(str(full_path), **local_args)
        else:
            logger = logging.getLogger(f"download.{obj_type}")
            logger.info(f"从Hugging Face下载: {name}")
            # 确保目录存在
            full_path.parent.mkdir(parents=True, exist_ok=True)
            # 传递远程参数
            obj = load_func(name, **remote_args)
            save_func(obj, str(full_path))
            logger.info(f"已保存至: {full_path}")
            
        logger.info("加载完成")
        return obj
        
    except Exception as e:
        logger = logging.getLogger(f"error.{obj_type}")
        logger.error(f"加载失败: {str(e)}", exc_info=True)
        return None


class Util:
    """Hugging Face资源加载工具类"""
    
    @staticmethod
    def load_or_download_dataset(
        name: str,
        split: Optional[str] = None,
        path: Union[str, Path] = dataset_save_path,
        **kwargs
    ) -> Optional[Union[Dataset, DatasetDict]]:
        """加载或下载数据集
        
        Args:
            name: 数据集名称或路径
            split: 数据集分割(如"train", "test"等)或切片语法(如"train[:50]")
            path: 本地保存路径
            **kwargs: 传递给load_dataset的其他参数
            
        Returns:
            加载的数据集对象，失败时返回None
        """
        path = Path(path) if isinstance(path, str) else path
        
        def parse_slice(s: str) -> tuple:
            # 解析类似"[:50]"的切片语法
            try:
                s = s.split("[")[1].rstrip("]")
                if ":" in s:
                    parts = s.split(":")
                    if len(parts) == 2:
                        start, end = parts
                        start = int(start) if start else 0
                        end = int(end) if end else None
                        return (start, end if end is not None else float('inf'))
                    elif len(parts) == 3:  # 处理step
                        start, end, step = parts
                        start = int(start) if start else 0
                        end = int(end) if end else None
                        step = int(step) if step else 1  # 处理step
                        # 简化处理，忽略step，只返回开始和结束索引
                        return (start, end if end is not None else float('inf'))
                return (0, int(s))
            except (ValueError, IndexError) as e:
                logger = logging.getLogger("error.parse_slice")
                logger.error(f"解析切片语法错误: {s}, {str(e)}")
                # 返回默认值
                return (0, float('inf'))
        
        # 创建远程加载处理函数
        def load_remote(source, **args):
            remote_kwargs = {**kwargs}
            if split:
                remote_kwargs['split'] = split
            return load_dataset(source, **remote_kwargs)
        
        # 创建本地加载处理函数
        def load_local(source, **args):
            try:
                ds = load_from_disk(source)
                if split:
                    if "[" in split:  # 处理切片语法
                        split_name = split.split("[")[0]
                        if split_name in ds:
                            return ds[split_name].select(range(*parse_slice(split)))
                        else:
                            logger = logging.getLogger("warning.dataset")
                            logger.warning(f"指定的split '{split_name}' 不存在于数据集中")
                            return ds  # 返回完整数据集
                    elif split in ds:  # 普通split
                        return ds[split]
                    else:
                        logger = logging.getLogger("warning.dataset")
                        logger.warning(f"指定的split '{split}' 不存在于数据集中")
                return ds
            except Exception as e:
                logger = logging.getLogger("error.dataset")
                logger.error(f"加载本地数据集失败: {str(e)}", exc_info=True)
                raise e
            
        return load_or_download(
            "数据集",
            load_func=load_local if (path / name).exists() else load_remote,
            save_func=lambda ds, p: ds.save_to_disk(p),
            name=name,
            path=path,
            local_args={},
            remote_args=kwargs
        )

    @staticmethod
    def load_or_download_tokenizer(
        name: str,
        path: Union[str, Path] = tokenizer_save_path,
        **kwargs
    ) -> Optional[PreTrainedTokenizer]:
        """加载或下载分词器
        
        Args:
            name: 分词器名称或路径
            path: 本地保存路径
            **kwargs: 传递给AutoTokenizer.from_pretrained的其他参数
            
        Returns:
            加载的分词器对象，失败时返回None
        """
        # 获取两种情况下要传递的参数
        local_args = kwargs.copy()
        # 从本地加载时可能需要移除一些参数，如默认就有的max_length
        if 'max_length' in local_args:
            del local_args['max_length']
        
        return load_or_download(
            "分词器",
            load_func=AutoTokenizer.from_pretrained,
            save_func=lambda tokenizer, p: tokenizer.save_pretrained(p),
            name=name,
            path=path,
            local_args=local_args,
            remote_args=kwargs
        )

    @staticmethod
    def load_or_download_model(
        name: str,
        path: Union[str, Path] = model_save_path,
        **kwargs
    ) -> Optional[PreTrainedModel]:
        """加载或下载模型
        
        Args:
            name: 模型名称或路径
            path: 本地保存路径
            **kwargs: 传递给AutoModelForCausalLM.from_pretrained的其他参数
            
        Returns:
            加载的模型对象，失败时返回None
        """
        # 保证trust_remote_code参数存在
        if 'trust_remote_code' not in kwargs:
            kwargs['trust_remote_code'] = True
            
        return load_or_download(
            "模型",
            load_func=AutoModelForCausalLM.from_pretrained,
            save_func=lambda model, p: model.save_pretrained(p),
            name=name,
            path=path,
            local_args=kwargs,  # 本地和远程加载使用相同参数
            remote_args=kwargs
        )
