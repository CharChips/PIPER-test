import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

class MyWebViewPage extends StatelessWidget {
  const MyWebViewPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Web View'),
      ),
      body: WebView(
        initialUrl: 'http://192.168.254.36/',
        javascriptMode: JavascriptMode.unrestricted,
      ),
    );
  }
}
