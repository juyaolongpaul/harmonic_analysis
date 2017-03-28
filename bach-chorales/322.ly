\version "2.10.33"

\header {
  title = "322 - Wenn mein Stündlein vorhanden ist"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \partial 4 g4 
    d4 e4 fis4 g4 
  a4 b4 g4 b8 c8 
  d4 d4 b8 c8 d4 
  c2 b4 b4 
  c4 b4 a4 g8 a8   %5
  b4 b8 a8 g4 b8 c8 
  d4 d4 b4 d4 
  c2 b4 b4 
  c4 b4 a4 g8 a8 
  b4 b4 g4 a4   %10
  b4 b4 e,4 fis4 
  g8 fis8 e4 d4 d4 
  g4 a4 b4 a8 b8 
  c4 b4 a2 
  g2.   %15
  
}

alto = \relative c' {
  \partial 4 d4 
    d4 c4 c4 d4 
  d4 d4 e4 d4 
  d8 e8 fis4 g4 f4 
  e4 fis4 d4 d4 
  d4. e8 fis4 g8 fis8   %5
  e4 dis4 e4 e4 
  d4 a'4 g4 b4~ 
  b4 a4 g4 g4 
  g4 g4 fis8 e8 d4 
  d4 d4 e4 e8 fis8~   %10
  fis8 e16 fis16 g8 fis8 e4. d8 
  d4 d8 cis8 a4 d4 
  d4 d4 g4 g4 
  fis8 a4 g8 g4 fis4 
  d2.   %15
  
}

tenor = \relative c' {
  \partial 4 b8  a8 
    g4 g4 a4 b4 
  c4 b4 c4 b4 
  a4 a4 g8 a8 b4~ 
  b4 a4 g4 b4 
  a4 b4 b4 b4   %5
  b4 b4 b4 g4 
  a8 b8 c8 d8 d8 e8 f4 
  e4 fis4 d4 d4 
  c4 d4 d4 d8 c8 
  b8 a8 g4 c4 c4   %10
  b4 b4. a8 a8 b16 c16 
  b8 a8 g4 fis4 fis8 a8 
  b4 a4 g4 e'8 d8 
  c8 d8 d4 e4 d8 c8 
  b2.   %15
  
}

baixo = \relative c {
  \partial 4 g8  a8 
    b4 bes4 a4 g4 
  fis4 g4 c4 g'4 
  fis8 e8 d4 e4 b4 
  c4 d4 g,4 g'4 
  fis4 g4 dis4 e8 fis8   %5
  g8 a8 b4 e,4 e4 
  f4 fis4 g4 gis4 
  a4 d,4 g4 g8 fis8 
  e4 d4 c4 b8 a8 
  g8 a8 b4 c4 a'4   %10
  g8 fis8 e8 d8 cis4 d4 
  g,4 a4 d4 d8 c8 
  b8 a8 g8 fis8 e8 d'8 c8 b8 
  a8 fis8 g8 b8 c8 a8 d8 d,8 
  g2.   %15
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}