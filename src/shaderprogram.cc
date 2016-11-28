#include "shaderprogram.h"

#include <fstream>
#include <iostream>

#include <glbinding/gl/gl.h>

using namespace rscg;
using namespace gl;

//Check if shader has a error
static
std::string
CheckShaderError(GLuint shader, GLenum flag,
                 bool isProgram);

//create a program and return his ID
static
GLuint
CreateShader(const std::string &programSrc, GLenum shaderType);


//load shader source file
static
std::string
LoadShader(const std::string &fileName);

ShaderProgram::ShaderProgram(const std::string resPath,
                             const std::vector<GLenum> &shadersKind) :
                             ShaderProgram(std::vector<std::string>
                               (shadersKind.size(), resPath), shadersKind)
{
}

ShaderProgram::ShaderProgram(const std::vector<std::string> &pathList,
                             const std::vector<GLenum> &shadersKind)
{
  load(pathList, shadersKind);
}

ShaderProgram::ShaderProgram()
{
}

void
ShaderProgram::load(const std::string resPath,
                    const std::vector<GLenum> &shadersKind)
{
  load(std::vector<std::string>(shadersKind.size(), resPath), shadersKind);
}

void
ShaderProgram::load(const std::vector<std::string> &pathList,
                    const std::vector<GLenum> &shadersKind)
{
  shadersList.resize(shadersKind.size());

  std::vector<GLenum> kinds({GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER,
        GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER});

  std::vector<std::string> extNames({"shader.vert.glsl", "shader.tesscont.glsl",
        "shader.tesseval.glsl", "shader.geom.glsl", "shader.frag.glsl"});

  _program = glCreateProgram();
  int currentPosInPath = 0;
  for(int currentShaderType = 0;currentShaderType < kinds.size();
      ++currentShaderType)
  {
    for(auto mineShaders : shadersKind)
    {
      if(mineShaders == kinds[currentShaderType])
      {
        shadersList[currentPosInPath] =
          CreateShader(LoadShader(pathList[currentPosInPath]+ "/" +
                                  extNames[currentShaderType]),
                       mineShaders);
        ++currentPosInPath;
      }
    }
  }

  for(GLuint &i : shadersList)
    glAttachShader(_program, i);

  glLinkProgram(_program);
  auto strError = CheckShaderError(_program, GL_LINK_STATUS, true);

  glValidateProgram(_program);
  strError += CheckShaderError(_program, GL_VALIDATE_STATUS, true);

  if(strError.length() > 5)
    throw std::runtime_error(strError);
}

ShaderProgram::~ShaderProgram()
{
  for(GLuint &i: shadersList)
  {
    glDetachShader(_program, i);
    glDeleteShader(i);
  }

  glDeleteProgram(_program);
}

inline GLint
ShaderProgram::uniformLocation(std::string name)
{
  return glGetUniformLocation(_program, name.c_str());
}

void
ShaderProgram::bind() const
{
  glUseProgram(_program);
}

GLuint
ShaderProgram::programID() const
{
  return _program;
}

void
ShaderProgram::updateUniform(const glm::mat4 &value, const std::string &name)
{
  auto loc = uniformLocation(name);
  bind();
}

static
GLuint
CreateShader(const std::string &programSrc, GLenum shaderType)
{
  GLuint shader = glCreateShader(shaderType);

  if(shader == 0)
    std::cerr << "erro at creating shader" << std::endl;

  const GLchar *shaderCstrSrc = programSrc.c_str();
  const GLint programLength = static_cast<GLint>(programSrc.length());

  glShaderSource(shader, 1, &shaderCstrSrc, &programLength);
  glCompileShader(shader);

  auto status = CheckShaderError(shader, GL_COMPILE_STATUS, false);
  if(status.length() > 5)
    throw std::runtime_error(status);

  return shader;
}

static
std::string
CheckShaderError(GLuint shader, GLenum flag,
                 bool isProgram)
{
  GLboolean sucess = 0;
  GLchar error[1024] = {0};

  if(isProgram)
    glGetProgramiv(shader, flag, &sucess);
  else
    glGetShaderiv(shader, flag, &sucess);

  if(sucess == GL_FALSE)
  {
    if(isProgram)
      glGetProgramInfoLog(shader, sizeof(error), nullptr, error);
    else
      glGetShaderInfoLog(shader, sizeof(error), nullptr, error);
  }

  return std::string(error);
}

static
std::string
LoadShader(const std::string &resourcePath)
{
  std::ifstream fileStream;
  std::string content;

  fileStream.open(resourcePath.c_str());

  if(! fileStream.good())
    throw std::invalid_argument("resource Path:" + resourcePath);

  std::string line;

  for(;! fileStream.eof(); content += '\n')
  {
    std::getline(fileStream, line);

    content += line;
  }

  fileStream.close();

  return content;
}
