\version "2.10.33"

\header {
  title = "117 - O Welt, ich muß dich lassen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key aes \major
}


soprano = \relative c'' {
  \partial 4 c4 
    aes4 bes4 c8 des8 ees4 
  des2 c4 c8 des8 
  ees4 ees4 bes4 c4 
  aes2 g4 ees4 
  aes4 bes4 c4 c4   %5
  bes2. c4 
  aes4 bes4 c8 des8 ees4 
  des2 c4 c8 des8 
  ees4 ees4 bes4 c4 
  aes2 g4 ees4   %10
  aes4 bes4 c4 des4 
  c4 bes4 aes2 
  
}

alto = \relative c'' {
  \partial 4 aes4 
    f4 g4 aes8 bes8 c4 
  c4 bes4 aes4 aes4 
  aes8 g8 aes8 f8 g8 f8 ees4 
  ees4 d4 ees4 bes4 
  ees4 des4 c8 des8 ees4   %5
  ees2. g4 
  f4 ees4 ees4 f4 
  f8 ees8 f8 g8 a4 aes4 
  bes4 aes8 g8 f4 g4 
  g4 f4 e4 ees4   %10
  ees8 f8 ees8 des8 c8 ees8 f8 g8 
  aes4 g4 ees2 
  
}

tenor = \relative c' {
  \partial 4 ees4 
    des4 des4 ees8 des8 c8 bes8 
  aes4 ees'4 ees4 f4 
  ees4 ees4 ees4 g,4 
  aes8 g8 aes4 bes4 g8 f8 
  ees4 f8 g8 aes4 aes4   %5
  g2. c4 
  c8 des8 c8 bes8 aes8 bes8 c4 
  bes8 c8 des8 ees8 f4 f4 
  ees8 des8 c4 d4 g,4 
  c2 c4 c4   %10
  c8 des8 c8 bes8 aes4 aes4 
  f'4 bes,8 c16 des16 c2 
  
}

baixo = \relative c' {
  \partial 4 aes4 
    des8 c8 bes4 aes4. g8 
  f4 g4 aes4 f4 
  c8 bes8 c8 des8 ees8 des8 c4 
  f2 ees4 des4 
  c4 bes4 aes8 bes8 c8 des8   %5
  ees2. e4 
  f4 g4 aes4 a4 
  bes4 bes,4 f'4 f4 
  g4 aes4 aes8 g8 f8 e8 
  f8 g8 aes8 bes8 c4 c,4   %10
  f4 g4 aes8 g8 f8 ees8 
  des4 ees4 aes,2 
  
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