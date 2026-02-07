import { app } from "../../scripts/app.js"
import { ComfyWidgets } from "../../scripts/widgets.js"

app.registerExtension({
    name: "godmt.ListUtils",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Apt_UnitPromptWeight") {
            const onNodeCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined
                this.showValueWidget = ComfyWidgets["STRING"](this, "prompt_info", ["STRING", { multiline: true }], app).widget
            }
            const onExecuted = nodeType.prototype.onExecuted
            nodeType.prototype.onExecuted = function (message) {
                onExecuted === null || onExecuted === void 0 ? void 0 : onExecuted.apply(this, [message])
                if (message.text && message.text.length > 0) {
                    this.showValueWidget.value = message.text[0]
                }
            }
        }
    }
})
