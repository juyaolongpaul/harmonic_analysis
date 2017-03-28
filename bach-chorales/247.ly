\version "2.10.33"

\header {
  title = "247 - Wenn wir in höchsten Nöten sein"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \partial 4 g4 
  g4 a4 b4 a8 b8 
  c8 b8 a4 g4 b4 
  c4 b4 a4 g4 
  fis8 g8 g8. fis32 g32 a4 d4 
  c4 b4 a4 g8 fis8   %5
  e8. fis32 g32 e4 d4 b'4 
  c4 b4 a4 g8 a8 
  b4 a g
  
}

alto = \relative c' {
  \partial 4 d4 
  e4. d8 d8 e8 fis4 
  e8 g16 fis16 e8 fis8 d4 g4 
  g8 a4 g8 fis8 e16 dis16 e4 
  d4 d4 d4 fis4 
  e4 d8 cis16 d16 e8 fis16 e16 d4   %5
  d4 cis4 d4 fis4 
  e4 d4 e8 fis8 g4 
  g4 fis d
  
}

tenor = \relative c' {
  \partial 4 b4 
  b4 a8 fis8 g4 a4 
  a8 d8 c8 b16 a16 b4 d4 
  e8 dis4 e8 c4 b4 
  a4 b4 fis4 b8 a8 
  g8 a8 b4 cis4 b4   %5
  b8 g8 a4 fis4 fis4 
  g8 a8 b4 e,8 b'4 c8 
  d4 c16 b16 c8 b4
  
}

baixo = \relative c' {
  \partial 4 g8  fis8 
  e8 d8 c8 d8 g,4 d'4 
  a8 b8 c8 d8 g,4 g'8 fis8 
  e4 e4 e4. d8 
  c8 b8 a8 g8 d'4 b4 
  e8 fis8 g4. fis8 b8 a8   %5
  g8 e8 a4 d,4 dis4 
  e8 fis8 g4 cis,8 dis8 e8 d16 c16 
  b8 c8 d4 g,
  
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