"""统一工具处理器"""

import json
from typing import Dict, List, Any, Optional
from config.logger import setup_logging
from plugins_func.loadplugins import auto_import_modules

from .base import ToolType
from .server_plugins import ServerPluginExecutor
from plugins_func.register import Action, ActionResponse
from .unified_tool_manager import ToolManager

class UnifiedToolHandler:
    """统一工具处理器"""

    def __init__(self, conn):
        self.conn = conn
        self.config = conn.config
        self.logger = setup_logging()

        self.tool_manager = ToolManager(conn)

        # 创建各类执行器
        self.server_plugin_executor = ServerPluginExecutor(conn)
        
        self.tool_manager.register_executor(
            ToolType.SERVER_PLUGIN, self.server_plugin_executor
        )

        # 初始化标志
        self.finish_init = False

    async def _initialize(self):
        """异步初始化"""
        try:
            # 自动导入插件模块
            auto_import_modules("plugins_func.functions")

            self.finish_init = True
            self.logger.info("统一工具处理器初始化完成")
            self.current_support_functions()

        except Exception as e:
            self.logger.error(f"统一工具处理器初始化失败: {e}")


    def get_functions(self) -> List[Dict[str, Any]]:
        """获取所有工具的函数描述"""
        return self.tool_manager.get_function_descriptions()

    def current_support_functions(self) -> List[str]:
        """获取当前支持的函数名称列表"""
        func_names = self.tool_manager.get_supported_tool_names()
        self.logger.info(f"当前支持的函数列表: {func_names}")
        return func_names

    def upload_functions_desc(self):
        """刷新函数描述列表"""
        self.tool_manager.refresh_tools()
        self.logger.info("函数描述列表已刷新")

    def has_tool(self, tool_name: str) -> bool:
        """检查是否有指定工具"""
        return self.tool_manager.has_tool(tool_name)

    async def handle_llm_function_call(
        self, conn, function_call_data: Dict[str, Any]
    ) -> Optional[ActionResponse]:
        """处理LLM函数调用"""
        try:
            # 处理多函数调用
            if "function_calls" in function_call_data:
                responses = []
                print(f"处理多个函数调用: {function_call_data['function_calls']}")
                for call in function_call_data["function_calls"]:
                    result = await self.tool_manager.execute_tool(
                        call["name"], call.get("arguments", {})
                    )
                    responses.append(result)
                return self._combine_responses(responses)

            # 处理单函数调用
            function_name = function_call_data["name"]
            arguments = function_call_data.get("arguments", {})

            # 如果arguments是字符串，尝试解析为JSON
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    self.logger.error(f"无法解析函数参数: {arguments}")
                    return ActionResponse(
                        action=Action.ERROR,
                        response="无法解析函数参数",
                    )

            self.logger.info(f"调用函数: {function_name}, 参数: {arguments}")

            # 执行工具调用
            result = await self.tool_manager.execute_tool(function_name, arguments)
            return result

        except Exception as e:
            self.logger.error(f"处理function call错误: {e}")
            return ActionResponse(action=Action.ERROR, response=str(e))

    def _combine_responses(self, responses: List[ActionResponse]) -> ActionResponse:
        """合并多个函数调用的响应"""
        if not responses:
            return ActionResponse(action=Action.NONE, response="无响应")

        # 如果有任何错误，返回第一个错误
        for response in responses:
            if response.action == Action.ERROR:
                return response

        # 合并所有成功的响应
        contents = []
        responses_text = []

        for response in responses:
            if response.content:
                contents.append(response.content)
            if response.response:
                responses_text.append(response.response)

        # 确定最终的动作类型
        final_action = Action.RESPONSE
        for response in responses:
            if response.action == Action.REQLLM:
                final_action = Action.REQLLM
                break

        return ActionResponse(
            action=final_action,
            result="; ".join(contents) if contents else None,
            response="; ".join(responses_text) if responses_text else None,
        )

    def get_tool_statistics(self) -> Dict[str, int]:
        """获取工具统计信息"""
        return self.tool_manager.get_tool_statistics()

    async def cleanup(self):
        """清理资源"""
        try:
            await self.server_mcp_executor.cleanup()

            # 清理MCP接入点连接
            if (
                hasattr(self.conn, "mcp_endpoint_client")
                and self.conn.mcp_endpoint_client
            ):
                await self.conn.mcp_endpoint_client.close()

            self.logger.info("工具处理器清理完成")
        except Exception as e:
            self.logger.error(f"工具处理器清理失败: {e}")
