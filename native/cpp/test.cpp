#include <napi.h>


Napi::String Hello(
    const Napi::CallbackInfo& info
)
{
    Napi::Env env = info.Env();

    return Napi::String::New(
        env,
        "hello napi"
    );
}


Napi::Object Init(
    Napi::Env env,
    Napi::Object exports
)
{
    exports.Set(
        "hello",
        Napi::Function::New(
            env,
            Hello
        )
    );

    return exports;
}


NODE_API_MODULE(
    addon,
    Init
)