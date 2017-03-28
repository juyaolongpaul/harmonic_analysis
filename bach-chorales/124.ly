\version "2.10.33"

\header {
  title = "124 - Auf, auf, mein Herz, und du, mein ganzer Sinn"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \partial 4 b4 
    b4 b4 b4 e4 
  d4 d4 c4 c4 
  b4 g4 a4 a4 
  b4 c4 d8 c8 b4 
  a4 a4 g4 a4   %5
  a4 a4 a4 b8 a8 
  g4 g4 a4 g4 
  fis2 fis4 fis4 
  g4 g4 a4 a4 
  b4 c4 d8 c8 b4   %10
  a2 g2 
  
}

alto = \relative c'' {
  \partial 4 g4 
    g8 e8 b'8 a8 g4 a8 g8 
  fis8 e8 fis8 gis8 a4 e8 fis8 
  g4 d4 d8 e8 fis8 e8 
  d4 g8 e8 fis8 a8 g4~ 
  g8 fis16 e16 fis4 d4 e4   %5
  fis8 g8 fis8 e8 fis4 fis4~ 
  fis4 e4. fis8 b,8 e8~ 
  e8 dis8 e8 cis8 dis4 b4 
  b4 e8 d8 e8 fis16 g16 a8 fis8 
  d8 e16 fis16 g8 fis16 e16 fis8 a4 g8   %10
  g4 fis4 d2 
  
}

tenor = \relative c' {
  \partial 4 d4 
    d8 e8 fis4 e4 e4 
  a,8 g8 a8 b8 c8 b8 a4 
  d4 b4 a4 d8 c8 
  b8 a8 g4 a8 fis8 g8 b8 
  e8 c8 a8 d16 c16 b4 cis4   %5
  d8 cis8 d8 e8 d4 b4 
  b4 b4 a8 fis8 g4 
  c4 cis4 b4 b8 a8 
  g8 fis8 g8 b8 c4. d16 c16 
  b4. a16 g16 a8 fis'8 d8 e8   %10
  e8 c8 a8 d16 c16 b2 
  
}

baixo = \relative c' {
  \partial 4 g8  fis8 
    g4 dis4 e4 c4 
  c4. b8 a8 gis8 a4 
  g4 g'4 fis8 e8 d4 
  g8 fis8 e4 d4 e8 d8 
  c8 a8 d4 g,4 g'4   %5
  fis8 e8 d8 cis8 d4 dis4 
  e8 fis8 g8 e8 cis8 dis8 e8 c8 
  a4 ais4 b4 dis4 
  e8 d8 c8 b8 a8 g8 fis8 d8 
  g8 g'8 e8 a8 fis8 d8 g8 e8   %10
  c8 a8 d4 g,2 
  
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