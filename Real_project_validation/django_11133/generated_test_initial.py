from django.http import HttpResponse
from django.test import SimpleTestCase


class HttpResponseMemoryviewTest(SimpleTestCase):
    def test_memoryview_content_returns_bytes(self):
        """memoryview content should return bytes representation of underlying data"""
        response = HttpResponse(memoryview(b"My Content"))
        self.assertEqual(response.content, b"My Content")

    def test_memoryview_empty_bytes(self):
        """memoryview wrapping empty bytes should return empty bytes"""
        response = HttpResponse(memoryview(b""))
        self.assertEqual(response.content, b"")

    def test_memoryview_content_multiple_access(self):
        """Repeated access to content should return same bytes"""
        response = HttpResponse(memoryview(b"Test Data"))
        first_access = response.content
        second_access = response.content
        self.assertEqual(first_access, second_access)
        self.assertEqual(first_access, b"Test Data")

    def test_memoryview_content_setter(self):
        """Setting content to a new memoryview should return bytes of new memoryview"""
        response = HttpResponse(memoryview(b"Initial"))
        response.content = memoryview(b"Updated")
        self.assertEqual(response.content, b"Updated")

    def test_memoryview_charset_handling(self):
        """memoryview content should not affect charset/encoding behavior"""
        response = HttpResponse(memoryview(b"Test"))
        # Should not raise UnicodeDecodeError or similar
        self.assertEqual(response.charset, 'utf-8')
        self.assertEqual(response['Content-Type'], 'text/html; charset=utf-8')

    def test_memoryview_iteration(self):
        """Iterating over response with memoryview should yield bytes chunks"""
        response = HttpResponse(memoryview(b"Hello World"))
        chunks = list(response)
        self.assertEqual(chunks, [b"Hello World"])

    def test_memoryview_content_length_header(self):
        """Content-Length header should reflect length of underlying bytes data"""
        data = b"Test Content"
        response = HttpResponse(memoryview(data))
        self.assertEqual(response['Content-Length'], str(len(data)))
        self.assertNotEqual(response['Content-Length'], str(len(str(memoryview(data)))))