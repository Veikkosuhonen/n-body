import { mat4, vec3 } from "webgpu-matrix";
import type { Vec2 } from "./types";

export class Camera {
  pan = vec3.create();
  zoom = 1.0;
  private _viewMatrix = mat4.create();

  getViewMatrix([w, h]: Vec2) {
    mat4.ortho(
        0,                   // left
        w,                   // right
        h,                   // bottom
        0,                   // top
        200,                 // near
        -200,                // far
        this._viewMatrix,    // dst
    ); 
    mat4.translate(this._viewMatrix, this.pan, this._viewMatrix);
    mat4.scale(this._viewMatrix, vec3.fromValues(this.zoom, this.zoom, 1), this._viewMatrix);
    return this._viewMatrix;
  }
}

