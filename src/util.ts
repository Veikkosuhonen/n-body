export const defineValues = (
  shaderSource: string,
  defines: Record<string, number>,
) => {
  let newShaderSource = shaderSource;
  Object.keys(defines).forEach((key) => {
    newShaderSource = newShaderSource.replaceAll(
      `<${key}>`,
      String(defines[key]),
    );
  });
  return newShaderSource;
};
