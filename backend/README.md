## Allowing the ONNX dylib to be used by onnxruntime_go

If you downloaded the ONNX Runtime library from the internet, macOS may have added a quarantine attribute to the file. This attribute triggers stricter security checks. Solve it with this:

```sh
# Recursively remove quarantine from the entire ONNX Runtime folder
xattr -r -d com.apple.quarantine /Users/t1000/Documents/apps/onnxruntime-osx-arm64-1.26.0/

# Verify the attribute is gone
xattr /Users/t1000/Documents/apps/onnxruntime-osx-arm64-1.26.0/lib/libonnxruntime.dylib
```