import { mat4 } from "./math";

export class OrbitCamera {
  theta = 0;
  phi = 0;
  distance = 5;
  target = [0, 0, 0];
  up = [0, 1, 0];

  getViewMatrix() {
    const eye = [
      this.target[0] +
        this.distance * Math.sin(this.phi) * Math.cos(this.theta),
      this.target[1] + this.distance * Math.cos(this.phi),
      this.target[2] +
        this.distance * Math.sin(this.phi) * Math.sin(this.theta),
    ];
    const viewMatrix = mat4.create();
    mat4.lookAt(viewMatrix, eye, this.target, this.up);
    return viewMatrix;
  }
}
