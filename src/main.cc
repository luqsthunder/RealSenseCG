#include <iostream>
#include <memory>

#include <SDL.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#endif

#include <glbinding\gl\gl.h>
#include <glbinding\Binding.h>

#ifndef _MSC_VER
#pragma warning(pop)
#endif


#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "realsenseimage.h"
#include "camera.h"
#include "shaderprogram.h"
#include "window.h"

int
main(int argc, char **argv)
{
  using namespace gl;

  rscg::Window window{1366, 768};

  glbinding::Binding::initialize();

  auto device = rscg::CameraDeviceWindows{};

  bool running = true;
  SDL_Event event;
  rscg::RealSenseImage depthImage{640, 480};

  rscg::ShaderProgram textureProgram{"Shaders/simpleshader",
                                    {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER}};
  rscg::ShaderProgram pointCloudProgram{"Shaders/SimplePointCloud",
                                       {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER}};

  glm::mat4 proj, view;
  proj = glm::perspective(60.f, 1366.f/768.f, 0.01f, 100.f);
  view = glm::lookAt(glm::vec3{0.f, 0.f, -1.f}, glm::vec3{0.f, 0.f, 1.f},
                     glm::vec3{0.f, 1.f, 0.f});

  while(running)
  {
    while(SDL_PollEvent(&event))
    {
      if((event.type == SDL_QUIT) || ((event.type == SDL_KEYUP) &&
                                       event.key.keysym.sym == SDLK_ESCAPE))
        running = false;
    }

    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(gl::GL_COLOR_BUFFER_BIT);

    depthImage.update(device);
    device.releaseFrame();
    depthImage.draw(textureProgram.programID());

    SDL_GL_SwapWindow(window);
  }
  return 0;
}
