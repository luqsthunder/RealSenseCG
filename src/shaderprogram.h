#ifndef __RSCG_SHADER_PROGRAM_H__
#define __RSCG_SHADER_PROGRAM_H__

#include <glbinding/gl/types.h>

#include <glm/mat4x4.hpp>

#include <vector>
#include <string>

namespace rscg
{

/**
  * class which load shaders and create program.
  * this class will load shaders, but shader name must be "shader" and file
  * extension  must follow these rules:
  *
  * vertex_shader                :vert.glsl
  * fragment_shaader             :frag.glsl
  * geometry_shader              :geom.glsl
  * tesselation_evaluation_shader:tesseval.glsl
  * tesselation_control_shader   :tesscont.glsl
  *
  * e.g: geometry shader will be name shader.geom.glsl
  */
class ShaderProgram
{
public:
/**
  * @brief creating shaders from path.
  * @param resPath path with containg shaders sources.
  * @param shadersKind enum with contain all kind of shaders you will may need.
  *
  * This constructor load all shader sources from path checking kinds of shader
  * you passing in shaders param, in e.g: resPath = shaders/scene1/water/,
  * and shadersKind = gl::GL_VERTEX_SHADER | gl::GL_TESS_CONTROL_SHADER |
  * gl::GL_TESS_EVALUATION_SHADER | gl::GL_FRAGMENT_SHADER, the constructor will take
  * all enum and search on resPath for all shaders with name shader.vert.glsl
  * for vertex shader, shader.tesseval.glsl for tesselation evaluation shader,
  * and so on.
  */
  ShaderProgram(const std::string resPath,
                const std::vector<gl::GLenum> &shadersKind);

/**
  * @brief crating each shader contained in shadersKind fallowing the vector
  * and opengl graphics pipepeline
  * @param pathList vector containing each path to every shader wich you
  * passed on shadersKind
  * @param shadersKind enum contaning all shaders kind you will may need.
  *
  * Everything like other constructor, but every shader are created
  * fallowing opengl pipeline and what shaders kind you passed, e.g:
  * shadersKind = GL_VERTEX_SHADER | GL_FRAGMENT_SHADER | GL_GEOMETRY_SHADER,
  * the constructor will fallow the first path on list to create the vertex
  * shader, the second path to create geometry shader, and at last fragment
  * shader. if there are any doubts about opengl pipeline google it ^^
  */
  ShaderProgram(const std::vector<std::string> &pathList,
                const std::vector<gl::GLenum> &shadersKind);

  ShaderProgram();

/**
  * Default destructor which will call everything will need to destry the
  * shaders.
  */
  virtual ~ShaderProgram();

  void load(const std::string resPath,
            const std::vector<gl::GLenum> &shadersKind);

  void load(const std::vector<std::string> &pathList,
            const std::vector<gl::GLenum> &shadersKind);

/**
  * @brief Bind the program to use
  */
  void bind() const;

/**
  * @brief get program ID
  *
  * @return program ID, GLuint
  */
  gl::GLuint programID() const;

/**
  * @param name name of location to return this own id.
  *
  * return the location id of a uniform variable of the shader
  */
  gl::GLint uniformLocation(std::string name);

/**
  * @param value value to be passed to uniform variabel on shader
  * @param name name of uniform variable
  *
  * @throw if name is incorrect std::runtime_error is throw
  */
  void updateUniform(const glm::mat4& value, const std::string& name);

  protected:
    std::vector<gl::GLuint> shadersList;
    gl::GLuint _program;
  };
}
#endif
